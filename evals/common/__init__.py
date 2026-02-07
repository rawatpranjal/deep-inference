"""
Common Test Utilities

Shared utilities, metrics, and cross-regime tests for evaluation suite.
"""

from .metrics import (
    COVERAGE_THRESHOLDS,
    RECOVERY_THRESHOLDS,
    AUTODIFF_THRESHOLDS,
    LAMBDA_THRESHOLDS,
    PSI_THRESHOLDS,
    check_metric,
    validate_coverage_run,
    validate_recovery_run,
    validate_autodiff_run,
    validate_lambda_run,
    validate_psi_run,
    format_validation_table,
)

__all__ = [
    "COVERAGE_THRESHOLDS",
    "RECOVERY_THRESHOLDS",
    "AUTODIFF_THRESHOLDS",
    "LAMBDA_THRESHOLDS",
    "PSI_THRESHOLDS",
    "check_metric",
    "validate_coverage_run",
    "validate_recovery_run",
    "validate_autodiff_run",
    "validate_lambda_run",
    "validate_psi_run",
    "format_validation_table",
]
