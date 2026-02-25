from pydantic import BaseModel, Field


class PaperCoordinates(BaseModel):
    """3D coordinates for a paper in embedding space."""

    paper_id: str = Field(description="Internal UUID of the paper")
    arxiv_id: str = Field(description="arXiv identifier")
    title: str
    coords: tuple[float, float, float] = Field(
        description="[x, y, z] coordinates from UMAP reduction"
    )
    cluster_id: int | None = Field(
        default=None, description="Cluster assignment (-1 or None for noise)"
    )
    chunk_count: int = Field(default=0, description="Number of chunks in this paper")


class Cluster(BaseModel):
    """Semantic cluster of papers."""

    id: int = Field(description="Cluster identifier")
    label: str = Field(description="Auto-generated cluster label from paper titles")
    paper_ids: list[str] = Field(description="List of paper IDs in this cluster")
