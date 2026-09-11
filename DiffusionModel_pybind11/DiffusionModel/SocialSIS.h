#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <numeric>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <memory>
#include <cstdint>
#include <type_traits>

using namespace std;

struct EdgeFast {           // 12 bytes — constant-threshold path
    int from_node;
    int to_node;
    int timestamp;
};

struct EdgeVar {            // 16 bytes — variable-threshold path
    int from_node;
    int to_node;
    int timestamp;
    uint32_t int_threshold;
};

// Convert a float threshold [0,1] to uint32 for integer comparison
inline uint32_t to_int_threshold(float t) {
    if (t >= 1.0f) return UINT32_MAX;
    if (t <= 0.0f) return 0;
    return (uint32_t)((double)t * 4294967296.0);
}

// Simple persistent thread pool
class ThreadPool {
    vector<thread> workers;
    queue<function<void()>> tasks;
    mutex mu;
    condition_variable cv;
    bool stop = false;
public:
    ThreadPool(int n) {
        for (int i = 0; i < n; i++) {
            workers.emplace_back([this] {
                for (;;) {
                    function<void()> task;
                    {
                        unique_lock<mutex> lk(mu);
                        cv.wait(lk, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    void enqueue(function<void()> f) {
        {
            lock_guard<mutex> lk(mu);
            tasks.push(move(f));
        }
        cv.notify_one();
    }
    ~ThreadPool() {
        {
            lock_guard<mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        for (auto& t : workers) t.join();
    }
    int size() const { return (int)workers.size(); }
};

class SocialSIS {
public:
    float default_threshold;
    uint32_t default_int_threshold;
    bool variable_threshold;
    unordered_set<int> nodes;
    vector<EdgeFast> edges_fast;
    vector<EdgeVar>  edges_var;
    int earliest_timestamp, latest_timestamp, dataset_length;
    double activation_length_percent;
    int activation_length;
    int num_nodes;
    int num_threads;
    ThreadPool pool;

    SocialSIS(string file_path, float threshold, double activation_length_percent, int activation_length)
        : default_threshold(threshold),
          variable_threshold(threshold == -1),
          activation_length_percent(activation_length_percent),
          activation_length(activation_length),
          num_threads(max(1u, thread::hardware_concurrency())),
          pool(num_threads)
    {
        default_int_threshold = variable_threshold ? 0 : to_int_threshold(threshold);

        ifstream infile(file_path);
        string line;
        size_t reserve_n = 0;
        if (FILE *fp = fopen(file_path.c_str(), "r")) {
            fseek(fp, 0, SEEK_END);
            reserve_n = ftell(fp) / 50;
            fclose(fp);
        }
        if (variable_threshold) edges_var.reserve(reserve_n);
        else                    edges_fast.reserve(reserve_n);

        if (variable_threshold) {
            while (getline(infile, line)) {
                int n1, n2, ts; float th;
                sscanf(line.c_str(), "%d %d %d %f", &n1, &n2, &ts, &th);
                nodes.insert(n1); nodes.insert(n2);
                edges_var.push_back({n1, n2, ts, to_int_threshold(th)});
            }
        } else {
            while (getline(infile, line)) {
                int n1, n2, ts;
                sscanf(line.c_str(), "%d %d %d", &n1, &n2, &ts);
                nodes.insert(n1); nodes.insert(n2);
                edges_fast.push_back({n1, n2, ts});
            }
        }
        infile.close();

        num_nodes = (int)nodes.size();

        if (activation_length == 0 && activation_length_percent == 0) {
            cout << "Both activation_length and activation_length_percent are 0" << endl;
            return;
        }
        if (activation_length != 0 && activation_length_percent != 0) {
            cout << "Both activation_length and activation_length_percent are not 0" << endl;
            return;
        }

        if (variable_threshold) {
            earliest_timestamp = edges_var.front().timestamp;
            latest_timestamp   = edges_var.back().timestamp;
        } else {
            earliest_timestamp = edges_fast.front().timestamp;
            latest_timestamp   = edges_fast.back().timestamp;
        }
        dataset_length = latest_timestamp - earliest_timestamp;

        if (activation_length == 0)
            activation_length = int(dataset_length * activation_length_percent);

        size_t n_edges = variable_threshold ? edges_var.size() : edges_fast.size();
        cout << "Dataset loaded. Nodes: " << num_nodes
             << ", edges: " << n_edges
             << ", time length: " << dataset_length
             << ", activation length: " << activation_length
             << ", threads: " << num_threads << endl;
    }

    int get_num_nodes() { return num_nodes; }

    vector<vector<int>> get_edges() {
        vector<vector<int>> out;
        if (variable_threshold) {
            out.reserve(edges_var.size());
            for (const auto &e : edges_var)
                out.push_back({e.from_node, e.to_node, e.timestamp});
        } else {
            out.reserve(edges_fast.size());
            for (const auto &e : edges_fast)
                out.push_back({e.from_node, e.to_node, e.timestamp});
        }
        return out;
    }

    struct PCG {
        uint64_t state, inc;
        PCG() {
            random_device rd;
            state = ((uint64_t)rd() << 32) | rd();
            inc = (((uint64_t)rd() << 32) | rd()) | 1ULL;
        }
        uint32_t next() {
            uint64_t old = state;
            state = old * 6364136223846793005ULL + inc;
            uint32_t xs = ((old >> 18u) ^ old) >> 27u;
            uint32_t rot = old >> 59u;
            return (xs >> rot) | (xs << ((-rot) & 31));
        }
    };

    // Single Monte Carlo simulation run, specialized by edge type + normalization mode
    template <bool NormalizeByDataset, typename EdgeT>
    float cal_impl(const vector<int>& t_seeds,
                   const vector<EdgeT>& filtered_edges,
                   uint32_t global_int_threshold)
    {
        thread_local PCG rng;
        thread_local vector<int>     end_time;
        thread_local vector<uint8_t> is_activated;
        thread_local vector<int>     activated_nodes;

        if ((int)end_time.size() != num_nodes) {
            end_time.assign(num_nodes, -1);
            is_activated.assign(num_nodes, 0);
        }

        activated_nodes.clear();
        float total_length = 0;

        for (int seed : t_seeds) {
            end_time[seed] = latest_timestamp;
            is_activated[seed] = 1;
            activated_nodes.push_back(seed);
            total_length += latest_timestamp - earliest_timestamp;
        }

        constexpr bool VarT = is_same_v<EdgeT, EdgeVar>;

        for (const EdgeT &edge : filtered_edges) {
            if (!is_activated[edge.from_node]) continue;
            if (edge.timestamp > end_time[edge.from_node]) continue;

            uint32_t th;
            if constexpr (VarT) th = edge.int_threshold;
            else                th = global_int_threshold;

            if (rng.next() < th) {
                int &to_end = end_time[edge.to_node];
                if (!is_activated[edge.to_node]) {
                    is_activated[edge.to_node] = 1;
                    activated_nodes.push_back(edge.to_node);
                    int new_end = min(edge.timestamp + activation_length, latest_timestamp);
                    total_length += new_end - edge.timestamp;
                    to_end = new_end;
                } else if (edge.timestamp <= to_end) {
                    int new_end = min(to_end + activation_length, latest_timestamp);
                    total_length += new_end - to_end;
                    to_end = new_end;
                } else {
                    int new_end = min(edge.timestamp + activation_length, latest_timestamp);
                    total_length += new_end - edge.timestamp;
                    to_end = new_end;
                }
            }
        }

        for (int n : activated_nodes) {
            end_time[n] = -1;
            is_activated[n] = 0;
        }

        if constexpr (NormalizeByDataset)
            return total_length / dataset_length / num_nodes;
        else
            return total_length / num_nodes;
    }

    template <bool NormalizeByDataset>
    double influence_impl(const vector<int>& seeds_index, int num_repeat)
    {
        for (int seed : seeds_index)
            if (nodes.find(seed) == nodes.end()) return -1;

        int base = num_repeat / num_threads;
        int remainder = num_repeat % num_threads;

        unordered_set<int> seeds_set(seeds_index.begin(), seeds_index.end());

        vector<double> outputs(num_repeat);
        int remaining = num_threads;
        mutex mu;
        condition_variable cv;

        auto launch_tasks = [&](auto filtered_ptr, uint32_t gt, auto edge_tag) {
            using EdgeT = typename decltype(edge_tag)::type;
            int start = 0;
            for (int i = 0; i < num_threads; i++) {
                int count = base + (i < remainder ? 1 : 0);
                pool.enqueue([this, &seeds_index, &outputs, &mu, &cv, &remaining,
                              start, count, filtered_ptr, gt] {
                    for (int j = 0; j < count; j++)
                        outputs[start + j] = cal_impl<NormalizeByDataset, EdgeT>(
                            seeds_index, *filtered_ptr, gt);
                    lock_guard<mutex> lk(mu);
                    if (--remaining == 0) cv.notify_one();
                });
                start += count;
            }
        };

        if (variable_threshold) {
            auto filtered = make_shared<vector<EdgeVar>>();
            filtered->reserve(edges_var.size());
            for (const auto &e : edges_var)
                if (!seeds_set.count(e.to_node)) filtered->push_back(e);
            struct Tag { using type = EdgeVar; };
            launch_tasks(filtered, 0u, Tag{});
        } else {
            auto filtered = make_shared<vector<EdgeFast>>();
            filtered->reserve(edges_fast.size());
            for (const auto &e : edges_fast)
                if (!seeds_set.count(e.to_node)) filtered->push_back(e);
            struct Tag { using type = EdgeFast; };
            launch_tasks(filtered, default_int_threshold, Tag{});
        }

        unique_lock<mutex> lk(mu);
        cv.wait(lk, [&] { return remaining == 0; });

        return accumulate(outputs.begin(), outputs.end(), 0.0) / outputs.size();
    }

    double influence(const vector<int>& seeds_index, int num_repeat) {
        return influence_impl<true>(seeds_index, num_repeat);
    }
    double influence_time(const vector<int>& seeds_index, int num_repeat) {
        return influence_impl<false>(seeds_index, num_repeat);
    }
};