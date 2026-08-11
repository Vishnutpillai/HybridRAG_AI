"""
Confidence scoring utilities for the RAG pipeline.

Important:
- RRF score is NOT a probability.
- Dense and BM25 scores are normalized before combining.
- The resulting confidence is a retrieval confidence estimate.
"""


def normalize_score(
    score: float,
    min_score: float,
    max_score: float,
) -> float:
    """
    Normalize a score to the range 0-1.

    Args:
        score: Score to normalize.
        min_score: Minimum score in the result set.
        max_score: Maximum score in the result set.

    Returns:
        Normalized score between 0 and 1.
    """

    if max_score == min_score:
        return 1.0

    normalized = (
        (score - min_score)
        / (max_score - min_score)
    )

    # Safety: keep value between 0 and 1
    return max(0.0, min(1.0, normalized))


def calculate_retrieval_confidence(
    dense_score: float,
    bm25_score: float,
    dense_scores: list[float],
    bm25_scores: list[float],
) -> float:
    """
    Calculate retrieval confidence using
    dense similarity and BM25 scores.

    Dense similarity weight: 60%
    BM25 keyword score weight: 40%

    Returns:
        Retrieval confidence between 0 and 1.
    """

    if not dense_scores or not bm25_scores:
        return 0.0

    dense_min = min(dense_scores)
    dense_max = max(dense_scores)

    bm25_min = min(bm25_scores)
    bm25_max = max(bm25_scores)

    normalized_dense = normalize_score(
        dense_score,
        dense_min,
        dense_max,
    )

    normalized_bm25 = normalize_score(
        bm25_score,
        bm25_min,
        bm25_max,
    )

    confidence = (
        0.6 * normalized_dense
        + 0.4 * normalized_bm25
    )

    return round(confidence, 4)


def calculate_overall_confidence(
    retrieval_confidence: float,
    evidence_confidence: float,
) -> float:
    """
    Calculate an overall confidence estimate.

    Retrieval confidence: 60%
    Evidence confidence: 40%
    """

    overall = (
        0.6 * retrieval_confidence
        + 0.4 * evidence_confidence
    )

    return round(overall, 4)


if __name__ == "__main__":

    print("=" * 60)
    print("CONFIDENCE MODULE TEST")
    print("=" * 60)

    dense_scores = [
        0.92,
        0.81,
        0.76,
        0.70,
        0.65,
    ]

    bm25_scores = [
        18.5,
        14.2,
        11.8,
        9.4,
        7.1,
    ]

    dense_score = dense_scores[0]
    bm25_score = bm25_scores[0]

    retrieval_confidence = calculate_retrieval_confidence(
        dense_score=dense_score,
        bm25_score=bm25_score,
        dense_scores=dense_scores,
        bm25_scores=bm25_scores,
    )

    evidence_confidence = 0.90

    overall_confidence = calculate_overall_confidence(
        retrieval_confidence=retrieval_confidence,
        evidence_confidence=evidence_confidence,
    )

    print(f"\nDense score: {dense_score}")
    print(f"BM25 score: {bm25_score}")

    print(
        f"\nRetrieval confidence: "
        f"{retrieval_confidence}"
    )

    print(
        f"Evidence confidence: "
        f"{evidence_confidence}"
    )

    print(
        f"Overall confidence: "
        f"{overall_confidence}"
    )

    print("\n✅ Confidence module working!")

