import pickle

with open('/home/ridvan/Documents/center_surround/exp_13_data_old_1_2_3_4/hyperparam_results.pkl', 'rb') as f:
    results = pickle.load(f)

print("="*50)
print("BEST HYPERPARAMETERS")
print("="*50)
for key, value in results['best_params'].items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")
print(f"\nBest validation correlation: {results['best_value']:.4f}")
print("="*50)

# Show top 5 trials
print("\nTop 5 trials:")
study = results['study']
trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
for i, trial in enumerate(trials):
    print(f"\n{i+1}. Trial {trial.number} - correlation: {trial.value:.4f}")
    for key, value in trial.params.items():
        if isinstance(value, float):
            print(f"     {key}: {value:.6f}")
        else:
            print(f"     {key}: {value}")