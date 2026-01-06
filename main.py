import torch
import os
import argparse
import ast

from scripts.baseline.baseline_tester import BaselineTester
from model_utils.models.learning.siamese import SiameseModelPairs
from scripts.evaluation.evaluator import Evaluator

device = "cuda" if torch.cuda.is_available() else "cpu"

# tester = BaselineTester(model_type="siglip", batch_size=1, device=device)
# a, b = tester.test(test_filepath="data/processed/validate_pairs_ref_10k.parquet")

# print(a)
# print(b)

# # 1. Build the SigLIP backbone exactly as training
# tester = BaselineTester(model_type="siglip", batch_size=1, device=device)
# backbone = tester.model_wrapper  # this exposes encode_text()

# # 2. Build the Siamese model
# model = SiameseModelPairs(
#     embedding_dim=768,
#     projection_dim=768,
#     backbone=backbone,
# ).to(device)

# # 3. Load your projector weights (placed in model_utils)
# state = torch.load("weights/best_model_siglip_pair.pt", map_location=device)
# model.load_state_dict(state)

# model.eval()

# emb = model.encode(["amazon", "amaz0n"])
# # print(emb.shape)  # (2, 768)
# cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
# # print(cos(emb[0:1], emb[1:2]))

def main():
    parser = argparse.ArgumentParser(description='VLM-based text similarity evaluation')
    parser.add_argument('--mode', type=str, 
                      choices=['baseline', 'evaluate_saved'], 
                      required=True,
                      help='Mode to run. "baseline" to test baseline models, "evaluate_saved" to evaluate a trained Siamese model')
    parser.add_argument('--test_filepath', type=str, required=True,
                      help='Path to test data (CSV or Parquet with fraudulent_name, real_name, label)')
    parser.add_argument('--baseline_model', type=str, choices=['clip', 'coca', 'flava', 'siglip', 'all'], default='clip',
                      help='Baseline model to test (for baseline mode)')
    parser.add_argument('--backbone', type=str, choices=['clip', 'coca', 'flava', 'siglip'], default='clip',
                help='Vision-language backbone to use in Siamese model (clip, siglip, flava, etc.)')
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

        elif args.baseline_model in ['clip', 'coca', 'flava', 'siglip']:
            print(f"Testing {args.baseline_model.upper()} baseline model...")
            tester = BaselineTester(model_type=args.baseline_model, batch_size=args.batch_size, device=args.device)
            _results_df, metrics = tester.test(args.test_filepath, plot_roc=args.plot_roc)
            print(f"\n{args.baseline_model.upper()} Baseline Results Summary:")
            metrics_to_print = {k: v for k, v in metrics.items() if k != 'roc_curve'}
            print(metrics_to_print)

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

if __name__ == '__main__':
    main() 
