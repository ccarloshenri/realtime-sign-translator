from src.models.libras_vocabulary import LIBRAS_BASE_VOCABULARY, LibrasSign


def test_vocabulary_size_and_uniqueness():
    assert len(LIBRAS_BASE_VOCABULARY) == len(set(LIBRAS_BASE_VOCABULARY))
    assert len(LIBRAS_BASE_VOCABULARY) == len(LibrasSign)


def test_vocabulary_matches_enum_values():
    assert LIBRAS_BASE_VOCABULARY == tuple(s.value for s in LibrasSign)


def test_from_label_is_case_and_accent_tolerant():
    assert LibrasSign.from_label("olá") is LibrasSign.OLA
    assert LibrasSign.from_label("  OLÁ  ") is LibrasSign.OLA
    assert LibrasSign.from_label("por favor") is LibrasSign.POR_FAVOR
    assert LibrasSign.from_label("bogus") is None
    assert LibrasSign.from_label("") is None
