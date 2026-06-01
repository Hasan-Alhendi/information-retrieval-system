"""Cluster documents use case."""


class ClusterDocumentsUseCase:
    """Use case for document clustering."""

    def execute(self, dataset_name: str, number_of_clusters: int = 10) -> dict[str, object]:
        """Cluster documents for a dataset.

        A full implementation will be added when clustering is migrated.
        """
        return {
            "dataset_name": dataset_name,
            "number_of_clusters": number_of_clusters,
            "clusters": [],
        }
