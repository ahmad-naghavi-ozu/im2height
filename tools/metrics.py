import numpy as np

def compute_dsm_metrics(
    verbose,
    logger,
    total_delta1,
    total_delta2,
    total_delta3,
    total_mse,
    total_mae,
    total_rmse,
    dsm_tile,
    dsm_pred
):
    """
    Compute Digital Surface Model (DSM) evaluation metrics for a single tile.
    This function calculates various error metrics between predicted and ground truth DSM tiles,
    including MSE, MAE, RMSE and delta accuracy metrics. It handles invalid values and updates
    running totals for batch processing.
    Args:
        verbose (bool): If True, logs detailed metrics for each tile
        logger: Logger object for output messages
        total_delta1 (float): Running total for delta1 accuracy metric
        total_delta2 (float): Running total for delta2 accuracy metric
        total_delta3 (float): Running total for delta3 accuracy metric
        total_mse (float): Running total for Mean Squared Error
        total_mae (float): Running total for Mean Absolute Error 
        total_rmse (float): Running total for Root Mean Squared Error
        dsm_tile (numpy.ndarray): Ground truth DSM tile
        dsm_pred (numpy.ndarray): Predicted DSM tile
    Returns:
        tuple:
            - total_delta1 (float): Updated running total for delta1
            - total_delta2 (float): Updated running total for delta2
            - total_delta3 (float): Updated running total for delta3
            - total_mse (float): Updated running total for MSE
            - total_mae (float): Updated running total for MAE
            - total_rmse (float): Updated running total for RMSE
            - dsm_tile (numpy.ndarray): Filtered ground truth DSM
            - dsm_pred (numpy.ndarray): Filtered predicted DSM
    Notes:
        - Delta metrics measure accuracy within thresholds of 1.25, 1.25^2, and 1.25^3
        - Zero or negative values are replaced with small positive values (1e-5)
        - Invalid pixels are filtered out before computation
    """
    # 1. Ensure both arrays have the same shape
    assert dsm_tile.shape == dsm_pred.shape, (
        f"Shape mismatch: dsm_tile shape {dsm_tile.shape}, "
        f"dsm_pred shape {dsm_pred.shape}"
    )

    # 2. Copy to avoid modifying the originals
    dsm_tile_ = dsm_tile.copy()
    dsm_pred_ = dsm_pred.copy()

    # 3. Handle invalid or zero values: the original approach based on the contest rules
    #    - Very small positive for zeros and negative in pred
    #    - Large sentinel (999) for negative pred -> Not used in this version!
    #    - Very small positive for non-positive ground truth

    # Calculate the average of zero and negative pixels for dsm_pred and dsm_tile
    zero_pixel_ratio_pred = len(dsm_pred_[dsm_pred_ == 0]) / dsm_pred_.size
    avg_neg_pred = np.mean(dsm_pred_[dsm_pred_ < 0])
    zero_pixel_ratio_tile = len(dsm_tile_[dsm_tile_ == 0]) / dsm_tile_.size
    avg_neg_tile = np.mean(dsm_tile_[dsm_tile_ < 0])
    
    # Log the ratios if verbose
    if verbose:
        logger.info(f"dsm_pred - ratio of zero pixels    : {zero_pixel_ratio_pred}")
        logger.info(f"dsm_pred - mean of negative pixels : {avg_neg_pred}")
        logger.info(f"dsm_tile - ratio of zero pixels    : {zero_pixel_ratio_tile}")
        logger.info(f"dsm_tile - mean of negative pixels : {avg_neg_tile}")

    # Replace zero or negative values to avoid division by zero or invalid ratios
    # dsm_pred_[dsm_pred_ == 0], dsm_pred_[dsm_pred_ < 0] = 1e-5, 999
    dsm_pred_[dsm_pred_ <= 0] = 1e-5
    dsm_tile_[dsm_tile_ <= 0] = 1e-5

    # 4. Create a valid mask (both arrays should have strictly positive values)
    valid_mask = (dsm_tile_ > 0) & (dsm_pred_ > 0)

    dsm_tile_ = dsm_tile_[valid_mask]
    dsm_pred_ = dsm_pred_[valid_mask]

    # 5. Check if there are any valid pixels left
    if len(dsm_tile_) == 0:
        if verbose:
            logger.warning("No valid pixels found in this tile!")
        return (
            total_delta1, total_delta2, total_delta3,
            total_mse, total_mae, total_rmse,
            dsm_tile_, dsm_pred_
        )
 
    # 6. Compute error metrics: MSE, MAE, RMSE
    abs_diff = np.abs(dsm_pred_ - dsm_tile_)
    tile_mse = np.mean(abs_diff ** 2)
    tile_mae = np.mean(abs_diff)
    tile_rmse = np.sqrt(tile_mse)

    # 7. Compute delta metrics
    #    max_ratio = max(pred / gt, gt / pred)
    max_ratio = np.maximum(dsm_pred_ / dsm_tile_, dsm_tile_ / dsm_pred_)
    tile_delta1 = np.mean(max_ratio < 1.25)
    tile_delta2 = np.mean(max_ratio < 1.25 ** 2)
    tile_delta3 = np.mean(max_ratio < 1.25 ** 3)

    # 8. Log tile-level metrics if verbose
    if verbose:
        logger.info(f"Tile MSE   : {tile_mse:.4f}")
        logger.info(f"Tile MAE   : {tile_mae:.4f}")
        logger.info(f"Tile RMSE  : {tile_rmse:.4f}")
        logger.info(f"Tile Delta1: {tile_delta1:.4f}")
        logger.info(f"Tile Delta2: {tile_delta2:.4f}")
        logger.info(f"Tile Delta3: {tile_delta3:.4f}")

    # 9. Update running totals
    total_mse  += tile_mse
    total_mae  += tile_mae
    total_rmse += tile_rmse

    total_delta1 += tile_delta1
    total_delta2 += tile_delta2
    total_delta3 += tile_delta3

    # 10. Return updated totals + filtered arrays
    return (
        total_delta1,
        total_delta2,
        total_delta3,
        total_mse,
        total_mae,
        total_rmse,
        dsm_tile_,
        dsm_pred_
    )
