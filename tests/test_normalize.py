from src.preprocessing.normalize import (
    ascii_fold,
    core_business_name,
    extract_emails,
    extract_house_number,
    extract_numbers,
    extract_phones,
    extract_postal_code,
    extract_region_hint,
    extract_urls,
    normalize_address,
    normalize_business_name,
    prefix_key,
    sorted_token_key,
    soundex_key,
)


def test_business_name_normalization_keeps_meaning():
    assert normalize_business_name("Acme Pvt. Ltd.") == "acme private limited"
    assert core_business_name("Acme Pvt. Ltd.") == "acme"


def test_address_normalization_expands_common_abbreviations():
    assert normalize_address("12 Main Rd., Apt 5") == "12 main road apartment 5"


def test_extract_numbers_and_postal_code():
    address = "1795 Westchester Drive, High Point, NC 27262"
    assert extract_numbers(address) == ["1795", "27262"]
    assert extract_postal_code(address, "US") == "27262"


def test_india_pin_code():
    assert extract_postal_code("MG Road, Bengaluru 560001", "India") == "560001"


def test_ascii_fold_preserves_canonical_unicode_path():
    assert ascii_fold("Café Déjà Vu") == "cafe deja vu"


def test_advanced_contact_extraction():
    text = "Call +1 (919) 555-1212 or info@Example.com, visit https://www.Example.com/store"
    assert extract_phones(text) == ["19195551212"]
    assert extract_emails(text) == ["info@example.com"]
    assert extract_urls(text) == ["example.com/store"]


def test_advanced_address_hints():
    address = "No. 12 Main Rd., Apt 5, High Point, NC 27262"
    assert extract_house_number(address) == "12"
    assert extract_region_hint(address, "US") == "NC"


def test_blocking_keys_are_deterministic():
    tokens = ["prime", "money", "prime"]
    assert sorted_token_key(tokens) == "money|prime"
    assert prefix_key(tokens) == "prim|mone|prim"
    assert soundex_key(tokens) == "P650|M500"
