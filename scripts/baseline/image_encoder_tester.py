"""
Image Encoder Tester - tests image encoders on glyph-based text similarity.

Mirrors BaselineTester structure but works with image encoders.
"""

import torch
from model_utils.models.image_encoder_factory import ImageEncoderFactory
from scripts.evaluation.image_encoder_evaluator import ImageEncoderEvaluator


class ImageEncoderTester:
    """
    Test image encoders on glyph-based text pair similarity tasks.
    
    Converts text to glyphs, encodes them using image encoders, and evaluates
    similarity metrics on test data. Mirrors BaselineTester interface.
    """
    
    def __init__(self, model_type, model_name=None, batch_size=32, device='cuda', glyph_size=(224, 224)):
        """
        Initialize image encoder tester.
        
        Args:
            model_type: Type of image encoder ('vit', 'resnet', 'convnext', 'vitmae', 'siglip')
            model_name: Custom model name (optional, uses default if not specified)
            batch_size: Batch size for encoding
            device: Device to use ('cuda' or 'cpu')
            glyph_size: Size of generated glyphs (width, height)
        """
        self.model_type = model_type
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.glyph_size = glyph_size
        
        # Create and load model
        self.model = ImageEncoderFactory.create_model(
            model_type=model_type,
            model_name=model_name,
            device=device
        )
        self.model.to(device)
        self.model.eval()
    
    def test(self, test_filepath, plot_roc=False):
        """
        Test a single image encoder on test data.
        
        Args:
            test_filepath: Path to test data (CSV or Parquet)
            plot_roc: Whether to plot ROC curve and confusion matrices
            
        Returns:
            tuple: (results_df, metrics) - Results and evaluation metrics
        """
        evaluator = ImageEncoderEvaluator(
            self.model,
            batch_size=self.batch_size,
            glyph_size=self.glyph_size
        )
        
        results_df, metrics = evaluator.test_pairs(test_filepath, plot=plot_roc)
        return results_df, metrics
    
    def test_all_models(self, test_filepath, plot_roc=False):
        """
        Test all available image encoder models.
        
        Args:
            test_filepath: Path to test data (CSV or Parquet)
            plot_roc: Whether to plot ROC curves for each model
            
        Returns:
            dict: Dictionary mapping model names to (results_df, metrics) tuples
        """
        results = {}
        
        # Get all available models
        registry = ImageEncoderFactory.list_models()
        
        for model_type in registry.keys():
            print(f"\n{'='*60}")
            print(f"Testing {model_type}")
            print(f"{'='*60}")
            
            try:
                tester = ImageEncoderTester(
                    model_type=model_type,
                    batch_size=self.batch_size,
                    device=self.device,
                    glyph_size=self.glyph_size
                )
                
                results_df, metrics = tester.test(test_filepath, plot_roc=plot_roc)
                results[model_type] = (results_df, metrics)
                
                print(f"\nResults for {model_type}:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
                print(f"  Threshold: {metrics['threshold']:.4f}")
                
            except Exception as e:
                print(f"Error testing {model_type}: {str(e)}")
                results[model_type] = None
        
        return results


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_encoder_tester.py <test_filepath> [model_type] [--device cuda]")
        sys.exit(1)
    
    test_filepath = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'vit'
    device = 'cuda' if '--device cuda' in sys.argv else 'cpu'
    
    tester = ImageEncoderTester(model_type=model_type, device=device)
    results_df, metrics = tester.test(test_filepath, plot_roc=True)
    
    print("\nResults Summary:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
