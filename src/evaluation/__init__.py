"""
Evaluation module for author switch detection.
Provides cross-validation, metrics, feature importance, and ablation studies.
"""

from evaluation.metrics import (
    cross_validate_model,
    evaluate_model,
    compute_metrics,
    print_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    find_optimal_threshold,
)

from evaluation.importance import (
    permutation_importance,
    logistic_coefficients,
    shap_summary,
    get_feature_importance_dataframe,
)

from evaluation.ablation import (
    run_leave_one_out,
    run_single_group,
    save_ablation_csv,
)