import unittest

from inference import CurrencyVerifier


class InferenceRegressionTests(unittest.TestCase):
    def test_supported_currency_classes_excludes_non_currency_buckets(self):
        verifier = object.__new__(CurrencyVerifier)
        verifier.class_names = ["1_Cedi_note", "_other", "2_Cedi_note"]
        verifier.stats = {"1_Cedi_note": {"threshold": 0.01}, "2_Cedi_note": {"threshold": 0.02}}

        self.assertTrue(verifier.is_supported_class("1_Cedi_note"))
        self.assertFalse(verifier.is_supported_class("_other"))
        self.assertTrue(verifier.is_supported_class("2_Cedi_note"))

    def test_low_confidence_prediction_is_rejected_before_anomaly_check(self):
        verifier = object.__new__(CurrencyVerifier)
        verifier.class_names = ["1_Cedi_note", "_other"]
        verifier.stats = {"1_Cedi_note": {"threshold": 0.01}}
        verifier.classifier_transform = None
        verifier.autoencoder_transform = None

        self.assertFalse(verifier.should_continue_with_prediction("1_Cedi_note", 0.4))
        self.assertTrue(verifier.should_continue_with_prediction("1_Cedi_note", 0.9))


if __name__ == "__main__":
    unittest.main()
