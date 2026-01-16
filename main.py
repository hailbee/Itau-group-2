import torch
import os
import argparse
import ast
import time
import pandas as pd

from scripts.baseline.baseline_tester import BaselineTester
from scripts.baseline.image_encoder_tester import ImageEncoderTester
from scripts.baseline.ocr_tester import OCRTester
from model_utils.models.learning.siamese import SiameseModelPairs
from scripts.evaluation.evaluator import Evaluator
from scripts.baseline.string_methods_tester import StringMethodTester

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description='VLM-based and Image Encoder text similarity evaluation')
    parser.add_argument('--mode', type=str, 
                      choices=['baseline', 'image_encoder', 'ocr', 'evaluate_saved', 'latency', 'string_methods'], 
                      required=True,
                      help='Mode to run. "baseline" to test VLM models, "image_encoder" to test image encoders on glyphs, "ocr" to test OCR on glyphs, "evaluate_saved" to evaluate a trained Siamese model, "latency" to measure VLM latency')
    parser.add_argument('--test_filepath', type=str, required=True,
                      help='Path to test data (CSV or Parquet with fraudulent_name, real_name, label)')
    parser.add_argument('--baseline_model', type=str, choices=['clip', 'coca', 'flava', 'siglip', 'all'], default='clip',
                      help='Baseline model to test (for baseline mode)')
    parser.add_argument('--image_encoder', type=str, 
                      choices=['vit', 'resnet', 'convnext', 'vitmae', 'siglip', 'all'], 
                      default='vit',
                      help='Image encoder model to test (for image_encoder mode)')
    parser.add_argument('--string_method_type', type=str, choices=['token-set', 'levenshtein'],
                        default='token-sort',
                        help='String method to test')
    parser.add_argument('--glyph_size', type=int, nargs=2, default=[224, 224],
                      help='Size of generated glyphs (width height)')
    parser.add_argument('--fuzzy_threshold', type=int, default=80,
                      help='OCR fuzzy matching threshold (0-100) for OCR mode')
    parser.add_argument('--ocr_thresholds', type=str, default='single',
                      choices=['single', 'all', 'custom'],
                      help='OCR threshold testing mode. "single" uses --fuzzy_threshold, "all" tests predefined thresholds, "custom" takes comma-separated values')
    parser.add_argument('--ocr_custom_thresholds', type=str, default=None,
                      help='Custom OCR thresholds as comma-separated values (e.g., "50,60,70,80")')
    parser.add_argument('--backbone', type=str, choices=['clip', 'coca', 'flava', 'siglip', 'cogvlm', 'qwenvlm', 'gemma'], default='clip',
                help='Vision-language backbone to use in Siamese model')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for testing')
    parser.add_argument('--plot_roc', type=ast.literal_eval, default=False, nargs='?',
                      help='Whether to plot ROC curve (True/False)')
    parser.add_argument('--model_weights', type=str, default=None,
                      help='Path to trained Siamese model weights (for evaluate_saved mode)')
    
    parser.add_argument('--device', type=str, default=device,
                      help='Device to use (cuda or cpu)')
    parser.add_argument('--plot', type=ast.literal_eval, default=False, nargs='?',
                      help='Whether to plot ROC and confusion matrix (True/False)')
    parser.add_argument('--save_misclassified', type=str, default=False,
                      help='Extract the misclassified samples as a csv')
    parser.add_argument('--latency_num_runs', type=int, default=10,
                      help='Number of inference runs for latency measurement')
    parser.add_argument('--latency_warmup', type=int, default=3,
                      help='Number of warmup runs for latency measurement')
    parser.add_argument('--measure_latency', type=ast.literal_eval, default=False, nargs='?',
                      help='Whether to measure and report latency for the selected mode')
    parser.add_argument('--include_preprocessing_time', type=ast.literal_eval, default=True, nargs='?',
                      help='Include preprocessing (text-to-glyph conversion) time in latency measurement')

    args = parser.parse_args()

    if args.mode == 'baseline':
        # Test baseline model(s) performance (pairwise only)
        if args.baseline_model == 'all':
            print("Testing all available baseline models...")
            tester = BaselineTester(model_type='clip', batch_size=args.batch_size, device=args.device) # model type doesn't matter here
            all_results = tester.test_all_models(args.test_filepath)
            print("\nBaseline Results Summary:")
            for model_type, result in all_results.items():
                if 'error' in result:
                    print(f"{model_type.upper()}: ERROR - {result['error']}")
                else:
                    metrics = result['metrics']
                    # Print only relevant metrics
                    metrics_to_print = {k: v for k, v in metrics.items() if k != 'roc_curve'}
                    print(f"{model_type.upper()}: {metrics_to_print}")

        elif args.baseline_model in ['clip', 'coca', 'flava', 'siglip', 'cogvlm', 'qwenvlm', 'gemma']:
            print(f"Testing {args.baseline_model.upper()} baseline model...")
            tester = BaselineTester(model_type=args.baseline_model, batch_size=args.batch_size, device=args.device)
            _results_df, metrics = tester.test(args.test_filepath, plot_roc=args.plot_roc)
            print(f"\n{args.baseline_model.upper()} Baseline Results Summary:")
            metrics_to_print = {k: v for k, v in metrics.items() if k != 'roc_curve'}
            print(metrics_to_print)

    elif args.mode == 'image_encoder':
        # Test image encoder(s) on glyph-based text similarity
        glyph_size = tuple(args.glyph_size)
        
        if args.image_encoder == 'all':
            print("Testing all available image encoders on glyphs...")
            tester = ImageEncoderTester(model_type='vit', batch_size=args.batch_size, device=args.device, glyph_size=glyph_size)
            all_results = tester.test_all_models(args.test_filepath, plot_roc=args.plot_roc)
            print("\nImage Encoder Results Summary:")
            for model_type, result in all_results.items():
                if result is None:
                    print(f"{model_type.upper()}: ERROR")
                else:
                    results_df, metrics = result
                    metrics_to_print = {k: v for k, v in metrics.items() if k != 'roc_curve'}
                    print(f"{model_type.upper()}: {metrics_to_print}")
        
        elif args.image_encoder in ['vit', 'resnet', 'convnext', 'vitmae', 'siglip']:
            print(f"Testing {args.image_encoder.upper()} image encoder on glyphs...")
            tester = ImageEncoderTester(model_type=args.image_encoder, batch_size=args.batch_size, device=args.device, glyph_size=glyph_size)
            _results_df, metrics = tester.test(args.test_filepath, plot_roc=args.plot_roc)
            print(f"\n{args.image_encoder.upper()} Image Encoder Results Summary:")
            metrics_to_print = {k: v for k, v in metrics.items() if k != 'roc_curve'}
            print(metrics_to_print)
        
        # Measure latency if requested
        if args.measure_latency:
            print("\n" + "="*60)
            print("Measuring image encoder latency...")
            print("="*60)
            from utils.latency_profiler import LatencyProfiler
            
            profiler = LatencyProfiler(num_runs=args.latency_num_runs, warmup_runs=args.latency_warmup)
            
            # Load sample texts for latency measurement
            if args.test_filepath.endswith('.csv'):
                df = pd.read_csv(args.test_filepath)
            else:
                df = pd.read_parquet(args.test_filepath)
            
            sample_texts = df['fraudulent_name'].astype(str).tolist()[:100]
            
            latency_metrics = profiler.profile_image_encoder(
                model=tester.model,
                test_texts=sample_texts,
                batch_size=args.batch_size,
                glyph_size=glyph_size,
                include_glyph_time=args.include_preprocessing_time
            )
            print(latency_metrics)

    elif args.mode == 'ocr':
        # Test OCR encoder on glyph-based text recognition
        glyph_size = tuple(args.glyph_size)
        
        print("Testing OCR encoder on glyphs...")
        tester = OCRTester(
            batch_size=args.batch_size,
            glyph_size=glyph_size,
            fuzzy_threshold=args.fuzzy_threshold
        )
        
        if args.ocr_thresholds == 'single':
            # Test single configuration
            print(f"Testing Pytesseract OCR with fuzzy threshold={args.fuzzy_threshold}...")
            results_df, metrics = tester.test(args.test_filepath, plot_roc=args.plot_roc)
            print(f"\nOCR Results Summary (fuzzy_threshold={args.fuzzy_threshold}):")
            metrics_to_print = {k: v for k, v in metrics.items() if k != 'roc_curve'}
            print(metrics_to_print)
            
        elif args.ocr_thresholds == 'all':
            # Test with predefined thresholds
            print("\nTesting OCR with predefined fuzzy thresholds...")
            all_results = tester.test_all_models(args.test_filepath, thresholds=None, plot_roc=args.plot_roc)
            print("\nOCR Results Summary (Predefined Thresholds):")
            for config_name, result in all_results.items():
                if result is None:
                    print(f"{config_name}: ERROR")
                else:
                    results_df, metrics = result
                    metrics_to_print = {k: v for k, v in metrics.items() if k != 'roc_curve'}
                    print(f"{config_name}: {metrics_to_print}")
        
        elif args.ocr_thresholds == 'custom':
            # Test with custom thresholds
            if args.ocr_custom_thresholds is None:
                raise ValueError("For 'custom' OCR thresholds, --ocr_custom_thresholds must be provided (e.g., '50,60,70,80')")
            
            try:
                custom_thresholds = [int(t.strip()) for t in args.ocr_custom_thresholds.split(',')]
            except ValueError:
                raise ValueError(f"Invalid custom thresholds format: {args.ocr_custom_thresholds}. Expected comma-separated integers.")
            
            print(f"\nTesting OCR with custom fuzzy thresholds: {custom_thresholds}...")
            all_results = tester.test_all_models(args.test_filepath, thresholds=custom_thresholds, plot_roc=args.plot_roc)
            print("\nOCR Results Summary (Custom Thresholds):")
            for config_name, result in all_results.items():
                if result is None:
                    print(f"{config_name}: ERROR")
                else:
                    results_df, metrics = result
                    metrics_to_print = {k: v for k, v in metrics.items() if k != 'roc_curve'}
                    print(f"{config_name}: {metrics_to_print}")
        
        # Measure latency if requested
        if args.measure_latency:
            print("\n" + "="*60)
            print("Measuring OCR latency...")
            print("="*60)
            from utils.latency_profiler import LatencyProfiler
            
            profiler = LatencyProfiler(num_runs=args.latency_num_runs, warmup_runs=args.latency_warmup)
            
            # Load sample texts for latency measurement
            if args.test_filepath.endswith('.csv'):
                df = pd.read_csv(args.test_filepath)
            else:
                df = pd.read_parquet(args.test_filepath)
            
            sample_texts = df['fraudulent_name'].astype(str).tolist()[:100]
            
            latency_metrics = profiler.profile_ocr(
                ocr_tester=tester,
                test_texts=sample_texts,
                include_glyph_time=args.include_preprocessing_time
            )
            print(latency_metrics)

    elif args.mode == 'evaluate_saved':
        if args.model_weights is None:
            raise ValueError("For 'evaluate_saved' mode, --model_weights must be provided.")
        print(f"Evaluating saved Siamese model with {args.backbone.upper()} backbone...")
        tester = BaselineTester(model_type=args.backbone, batch_size=args.batch_size, device=device)
        backbone_module = tester.model_wrapper
        siamese_model = SiameseModelPairs(
            embedding_dim=768,
            projection_dim=768,
            backbone=backbone_module,
        ).to(args.device)
        state = torch.load(args.model_weights, map_location=args.device)
        siamese_model.load_state_dict(state)
        siamese_model.eval()

        evaluator = Evaluator(siamese_model, batch_size=args.batch_size, model_type=args.backbone)
        results_df, metrics = evaluator.evaluate(args.test_filepath, plot=args.plot)
        print(f"\nSiamese Model Results Summary:")
        for k, v in metrics.items():
            if k != 'roc_curve':
                print(f"{k}: {v}")
        
        # Measure latency if requested
        if args.measure_latency:
            print("\n" + "="*60)
            print("Measuring Siamese model latency...")
            print("="*60)
            from utils.latency_profiler import LatencyProfiler
            
            profiler = LatencyProfiler(num_runs=args.latency_num_runs, warmup_runs=args.latency_warmup)
            
            # Load sample text pairs for latency measurement
            if args.test_filepath.endswith('.csv'):
                df = pd.read_csv(args.test_filepath)
            else:
                df = pd.read_parquet(args.test_filepath)
            
            sample_pairs = list(zip(
                df['fraudulent_name'].astype(str).tolist()[:100],
                df['real_name'].astype(str).tolist()[:100]
            ))
            
            latency_metrics = profiler.profile_siamese_model(
                siamese_model=siamese_model,
                test_pairs=sample_pairs,
                batch_size=args.batch_size,
                include_glyph_time=args.include_preprocessing_time
            )
            print(latency_metrics)
    
    elif args.mode == 'latency':
        # Measure latency of VLM baseline models
        from utils.latency_profiler import LatencyProfiler, create_latency_report
        
        profiler = LatencyProfiler(num_runs=args.latency_num_runs, warmup_runs=args.latency_warmup)
        
        # Load test data to get sample texts
        if args.test_filepath.endswith('.csv'):
            df = pd.read_csv(args.test_filepath)
        else:
            df = pd.read_parquet(args.test_filepath)
        
        sample_texts = df['fraudulent_name'].astype(str).tolist()[:100]
        
        print("Starting latency measurement for VLM baselines...")
        print(f"Number of runs: {args.latency_num_runs}")
        print(f"Warmup runs: {args.latency_warmup}")
        print(f"Sample texts: {len(sample_texts)}")
        print()
        
        results = []
        
        if args.baseline_model == 'all':
            model_types = ['clip', 'coca', 'flava', 'siglip']
        else:
            model_types = [args.baseline_model]
        
        for model_type in model_types:
            try:
                print(f"Profiling {model_type.upper()}...")
                tester = BaselineTester(model_type=model_type, batch_size=args.batch_size, device=args.device)
                metrics = profiler.profile_text_encoder(tester.model_wrapper, sample_texts, args.batch_size)
                results.append(metrics)
                print(metrics)
                print()
            except RuntimeError as e:
                error_msg = str(e)
                # Check if it's a gated model error
                if 'gated model' in error_msg.lower():
                    print(f"⚠️  {model_type.upper()} Skipped: Gated Model (Authentication Required)")
                    print(f"   → To use this model, visit: https://huggingface.co/google/gemma-2b")
                    print(f"   → Then run: huggingface-cli login")
                    print()
                # Check if it's a transformers library compatibility issue
                elif 'Unrecognized configuration class' in error_msg or 'Failed to create' in error_msg:
                    print(f"⚠️  {model_type.upper()} Skipped: {error_msg.split(':')[0]}")
                    if 'cogvlm' in model_type.lower() or 'qwen' in model_type.lower():
                        print(f"   → Tip: Try upgrading transformers: pip install --upgrade transformers")
                    print()
                else:
                    print(f"Error profiling {model_type}: {error_msg}")
                    print()
            except Exception as e:
                print(f"Error profiling {model_type}: {str(e)}")
                print()
        
        # Create and print report
        if results:
            report = create_latency_report(results)
            print(report)
            
            # Save report to file
            report_filename = f"latency_report_{int(time.time())}.txt"
            with open(report_filename, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {report_filename}")
    
    elif args.mode == 'string_methods':
        print(f"Testing string method: {args.string_method_type}")
        tester = StringMethodTester(type = args.string_method_type)
        
        metrics = tester.test(
            test_filepath=args.test_filepath,
            plot_roc=args.plot_roc
        )

        print("\nString Method Results Summary:")
        for k, v in metrics.items():
            if k != "roc_curve":
                print(f"{k}: {v}")
                
    # Save misclassified
    if args.save_misclassified:

        from scripts.evaluation.error_analysis import extract_misclassified

        misclassified_df = extract_misclassified(
            results_df,
            threshold=metrics["threshold"]
        )

        misclassified_df.to_csv(
            "misclassified_samples.csv",
            index=False
        )

        print(f"Saved {len(misclassified_df)} misclassified samples to misclassified_samples.csv")


if __name__ == '__main__':
    main() 

    # python Itau-group-2/main.py 
    # --mode evaluate_saved 
    # --test_filepath data/processed/validate_pairs_ref_10k.parquet 
    # --backbone siglip 
    # --model_weights weights\best_model_siglip_pair.pt 
    # --plot 

    # # Test single encoder
    # python main.py --mode image_encoder --image_encoder vit --test_filepath data/processed/validate_pairs_ref_10k.parquet 

    # # Test all encoders
    # python main.py --mode image_encoder --image_encoder all --test_filepath data/processed/validate_pairs_ref_10k.parquet

    # # Custom glyph size
    # python main.py --mode image_encoder --image_encoder vit --test_filepath data/processed/validate_pairs_ref_10k.parquet --glyph_size 256 256

    # # latency mode
    # python main.py --mode latency --baseline_model clip --test_filepath data/processed/validate_pairs_ref_10k.parquet

    # # Compare all 7 models:
    # python main.py --mode latency --baseline_model all --test_filepath data/processed/validate_pairs_ref_10k.parquet

    # find misclassified examples
    """
    python3 main.py \
    --mode evaluate_saved \
    --test_filepath data/processed/validate_pairs_ref_10k.parquet \
    --backbone siglip \
    --model_weights weights/best_model_siglip_pair.pt \
    --plot False \
    --save_misclassified True
    """

    # # Basic OCR test
    # python main.py --mode ocr --test_filepath data/processed/validate_pairs_ref_10k.parquet

    # # Test all fuzzy matching thresholds
    # python main.py --mode ocr --test_filepath data/processed/validate_pairs_ref_10k.parquet --ocr_thresholds all

    # # With visualization
    # python main.py --mode ocr --test_filepath data/processed/validate_pairs_ref_10k.parquet --plot_roc True

    ### Valerie

    # # Basic OCR test
    # python main.py --mode ocr --test_filepath data/processed/validate_pairs_ref_10k.parquet

    # # Siglip text encoder
    # python main.py --mode baseline --baseline_model siglip --test_filepath data/processed/validate_pairs_ref_10k.parquet

    # # Siglip image encoder
    # python main.py --mode image_encoder --image_encoder siglip --test_filepath data/processed/validate_pairs_ref_10k.parquet