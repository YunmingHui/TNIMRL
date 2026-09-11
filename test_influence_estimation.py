from DiffusionModel.diffusion_model import SocialSIS

diff_model = SocialSIS(edge_file_path='Data/Bitcoinalpha/Bitcoinalpha_edges_all.txt', edge_thresholds=0.5, activation_length_percent=0, activation_length=30*86400)

estimated_influence = diff_model.influence_time([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], repeat_times=10000)

print(estimated_influence)
