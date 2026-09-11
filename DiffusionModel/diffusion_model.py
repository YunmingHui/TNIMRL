import diffusion_model


class SocialSIS:
    def __init__(
        self,
        edge_file_path: str,
        edge_thresholds: float,
        activation_length_percent: float,
        activation_length: int,
    ):
        """
        Parameters:
        edge_file_path
        edge_thresholds:
            edge_thresholds=-1 -> read the threshold of each edge from the file;
            0<edge_thresholds<1 -> set the thresholds of all edges to edge_thresholds
        double activation_length_percent & int activation_length: two ways to set activation length
            default 0, only one should be 0
        """
        if (activation_length_percent == 0 and activation_length == 0) or (
            activation_length_percent != 0 and activation_length != 0
        ):
            raise Exception(
                "Only one of activation_length_percent and activation_length should be 0"
            )
        self.socialsis = diffusion_model.SocialSIS(
            edge_file_path,
            edge_thresholds,
            activation_length_percent,
            activation_length,
        )

    def get_number_nodes(self) -> int:
        return self.socialsis.get_num_nodes()

    def get_edges(self) -> list:
        return self.socialsis.get_edges()

    def influence(self, seeds_index: list, repeat_times: int) -> float:
        """
        Parameters:
        seeds_index: input of the seeds must be represented by their indexes obtained by function queries
        repeat_times: repeat how many times
        """
        return self.socialsis.influence(seeds_index, repeat_times)

    def influence_time(self, seeds_index: list, repeat_times: int) -> float:
        """
        same as influence but the return is time not percentage of time
        """
        return self.socialsis.influence_time(seeds_index, repeat_times)
