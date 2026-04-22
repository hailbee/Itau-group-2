from __future__ import annotations

import unittest

from domain_matcher import canonicalize_domain_host, normalize_domain_string


class DomainNormalizationTests(unittest.TestCase):
    def test_obfuscated_query_keeps_at_sign_in_host_and_normalized_stem(self) -> None:
        self.assertEqual(canonicalize_domain_host("f@cebook.com"), "f@cebook.com")
        self.assertEqual(normalize_domain_string("f@cebook.com"), "f@cebook")

    def test_url_userinfo_is_still_removed_for_real_urls(self) -> None:
        self.assertEqual(
            canonicalize_domain_host("https://user:pass@example.com/login?q=1"),
            "example.com",
        )
        self.assertEqual(
            canonicalize_domain_host("//user@example.com/path"),
            "example.com",
        )


if __name__ == "__main__":
    unittest.main()
