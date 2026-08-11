# ============================================================
# CONFIDENCE MODULE
# ============================================================


def _extract_result(result):
    """
    Extract document and retrieval scores from a hybrid result.

    Supports:
        1. Dictionary results
        2. Tuple/list results

    Expected dictionary format:

        {
            "document": document,
            "dense_score": 0.42,
            "bm25_score": 14.65,
            "rrf_score": 0.03
        }

    Expected tuple format:

        (document, dense_score, bm25_score)
    """

    # --------------------------------------------------------
    # DICTIONARY RESULT
    # --------------------------------------------------------

    if isinstance(result, dict):

        document = result.get("document")

        dense_score = result.get(
            "dense_score",
            0.0
        )

        bm25_score = result.get(
            "bm25_score",
            0.0
        )

        rrf_score = result.get(
            "rrf_score",
            0.0
        )

        return (
            document,
            float(dense_score or 0.0),
            float(bm25_score or 0.0),
            float(rrf_score or 0.0)
        )

    # --------------------------------------------------------
    # TUPLE / LIST RESULT
    # --------------------------------------------------------

    if isinstance(result, (tuple, list)):

        if len(result) >= 4:

            document = result[0]
            dense_score = result[1]
            bm25_score = result[2]
            rrf_score = result[3]

            return (
                document,
                float(dense_score or 0.0),
                float(bm25_score or 0.0),
                float(rrf_score or 0.0)
            )

        if len(result) == 3:

            document = result[0]
            dense_score = result[1]
            bm25_score = result[2]

            return (
                document,
                float(dense_score or 0.0),
                float(bm25_score or 0.0),
                0.0
            )

        if len(result) == 2:

            document = result[0]
            score = result[1]

            return (
                document,
                float(score or 0.0),
                0.0,
                0.0
            )

    raise ValueError(
        f"Unsupported hybrid result format: {type(result)}"
    )


# ============================================================
# DENSE SCORE NORMALIZATION
# ============================================================

def normalize_dense_score(score):
    """
    Normalize dense similarity score to 0-1.
    """

    score = float(score)

    return max(
        0.0,
        min(score, 1.0)
    )


# ============================================================
# BM25 SCORE NORMALIZATION
# ============================================================

def normalize_bm25_scores(scores):
    """
    Normalize BM25 scores relative to the
    strongest retrieved document.
    """

    if not scores:
        return []

    positive_scores = [
        max(float(score), 0.0)
        for score in scores
    ]

    max_score = max(
        positive_scores
    )

    if max_score <= 0:

        return [
            0.0
            for _ in positive_scores
        ]

    return [
        min(
            score / max_score,
            1.0
        )
        for score in positive_scores
    ]


# ============================================================
# RETRIEVAL CONFIDENCE
# ============================================================

def calculate_retrieval_confidence(results):
    """
    Calculate retrieval confidence.

    Uses:
        - Dense similarity
        - BM25 relevance

    Returns:
        float between 0 and 1
    """

    if not results:
        return 0.0

    dense_scores = []
    bm25_scores = []

    for result in results:

        (
            document,
            dense_score,
            bm25_score,
            rrf_score
        ) = _extract_result(result)

        dense_scores.append(
            normalize_dense_score(
                dense_score
            )
        )

        bm25_scores.append(
            max(
                float(bm25_score),
                0.0
            )
        )

    # --------------------------------------------------------
    # NORMALIZE BM25
    # --------------------------------------------------------

    normalized_bm25 = normalize_bm25_scores(
        bm25_scores
    )

    # --------------------------------------------------------
    # DENSE CONFIDENCE
    # --------------------------------------------------------

    dense_confidence = (
        sum(dense_scores)
        / len(dense_scores)
    )

    # --------------------------------------------------------
    # BM25 CONFIDENCE
    # --------------------------------------------------------

    if normalized_bm25:

        bm25_confidence = (
            sum(normalized_bm25)
            / len(normalized_bm25)
        )

    else:

        bm25_confidence = 0.0

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    retrieval_confidence = (
        0.5 * dense_confidence
        +
        0.5 * bm25_confidence
    )

    return max(
        0.0,
        min(
            retrieval_confidence,
            1.0
        )
    )


# ============================================================
# EVIDENCE CONFIDENCE
# ============================================================

def calculate_evidence_confidence(results):
    """
    Calculate evidence confidence from
    retrieved document content.
    """

    if not results:
        return 0.0

    valid_documents = []

    for result in results:

        (
            document,
            dense_score,
            bm25_score,
            rrf_score
        ) = _extract_result(result)

        if document is None:
            continue

        content = getattr(
            document,
            "page_content",
            ""
        )

        if not content:
            continue

        content = content.strip()

        if len(content) >= 50:

            valid_documents.append(
                content
            )

    if not valid_documents:
        return 0.0

    # --------------------------------------------------------
    # CHUNK COVERAGE
    # --------------------------------------------------------

    chunk_factor = min(
        len(valid_documents) / 5,
        1.0
    )

    # --------------------------------------------------------
    # CONTENT QUALITY
    # --------------------------------------------------------

    quality_scores = []

    for content in valid_documents:

        length = len(content)

        if length >= 500:

            quality = 1.0

        elif length >= 300:

            quality = 0.9

        elif length >= 200:

            quality = 0.8

        elif length >= 100:

            quality = 0.6

        else:

            quality = 0.4

        quality_scores.append(
            quality
        )

    text_quality = (
        sum(quality_scores)
        / len(quality_scores)
    )

    # --------------------------------------------------------
    # FINAL EVIDENCE CONFIDENCE
    # --------------------------------------------------------

    evidence_confidence = (
        0.5 * chunk_factor
        +
        0.5 * text_quality
    )

    return max(
        0.0,
        min(
            evidence_confidence,
            1.0
        )
    )


# ============================================================
# OVERALL CONFIDENCE
# ============================================================

def calculate_overall_confidence(
    retrieval_confidence,
    evidence_confidence
):
    """
    Combine retrieval and evidence confidence.

    Important:
    This is a heuristic score, NOT a calibrated probability.
    """

    retrieval_confidence = max(
        0.0,
        min(
            float(retrieval_confidence),
            1.0
        )
    )

    evidence_confidence = max(
        0.0,
        min(
            float(evidence_confidence),
            1.0
        )
    )

    # Retrieval is slightly more important.
    overall_confidence = (
        0.6 * retrieval_confidence
        +
        0.4 * evidence_confidence
    )

    return {
        "retrieval_confidence": round(
            retrieval_confidence,
            2
        ),

        "evidence_confidence": round(
            evidence_confidence,
            2
        ),

        "overall_confidence": round(
            overall_confidence,
            2
        )
    }


# ============================================================
# COMPLETE CONFIDENCE CALCULATION
# ============================================================

def calculate_confidence(results):
    """
    Calculate all confidence dimensions at once.

    Returns:

        {
            "retrieval_confidence": ...,
            "evidence_confidence": ...,
            "overall_confidence": ...
        }
    """

    retrieval_confidence = (
        calculate_retrieval_confidence(
            results
        )
    )

    evidence_confidence = (
        calculate_evidence_confidence(
            results
        )
    )

    return calculate_overall_confidence(
        retrieval_confidence,
        evidence_confidence
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("CONFIDENCE MODULE TEST")
    print("=" * 60)

    # Dummy retrieval values
    retrieval_confidence = 0.97
    evidence_confidence = 1.00

    confidence = calculate_overall_confidence(
        retrieval_confidence,
        evidence_confidence
    )

    print(
        f"\nRetrieval confidence: "
        f"{confidence['retrieval_confidence']:.2f}"
    )

    print(
        f"Evidence confidence: "
        f"{confidence['evidence_confidence']:.2f}"
    )

    print(
        f"Overall confidence: "
        f"{confidence['overall_confidence']:.2f}"
    )

    print("\n⚠️ Note:")
    print(
        "These are heuristic confidence scores, "
        "not calibrated probabilities."
    )

    print("\n✅ Confidence module working!")
    