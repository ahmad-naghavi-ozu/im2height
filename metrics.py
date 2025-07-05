import numpy as np

class HeightRegressionMetrics(object):
    def _init_(self):
        self.reset()

    def reset(self):
       
        
        self.mse = 0.0
        self.rmse = 0.0
        self.abs = 0.0
        self.rmse_building = 0.0
        self.rmse_matched = 0.0
        self.total_high_rise_rmse = 0.0
        self.total_mid_rise_rmse = 0.0
        self.total_low_rise_rmse = 0.0
        self.delta1_sum = 0.0
        self.delta2_sum = 0.0
        self.delta3_sum = 0.0
        self.total_samples = 0
        self.img_sample = 0
        self.count_mid_rise = 0
        self.count_high_rise = 0
        self.count_low_rise = 0
       

    def add_batch(self, gt_image, pre_image, gt_mask, pred_mask):
        assert gt_image.shape == pre_image.shape, "Shape mismatch: gt_image shape {}, pre_image shape {}".format(
            gt_image.shape, pre_image.shape)
        
        delta_gt_image = gt_image.copy()
        delta_pre_image = pre_image.copy()
        pre_image[pre_image <= 0] = 0.00001
        #pre_image[pre_image < 0] = 0.00001
        gt_image[gt_image <= 0] = 0.00001
        valid_mask = ((gt_image > 0) | (pre_image > 0))
        #print("Valid Mask", valid_mask.shape)
        building_mask = np.expand_dims((gt_mask == 0), axis=0)
        #print("Building Mask", building_mask.shape)
        matched_building_mask = np.expand_dims((gt_mask == 0) & (pred_mask == 0), axis=0)
        #print("Matched Building Mask", matched_building_mask.shape)
        if valid_mask.sum() > 0:
            
            mse_i = np.nanmean((gt_image[valid_mask] - pre_image[valid_mask]) ** 2)
            rmse_i = np.sqrt(mse_i)
            abs_i = np.nanmean(np.abs(gt_image[valid_mask] - pre_image[valid_mask]))

        if building_mask.sum() > 0:
            rmse_b = (np.nanmean((pre_image[building_mask] - gt_image[building_mask]) ** 2)) ** 0.5
        else:
            rmse_b = 0.0

        if matched_building_mask.sum() > 0:
            rmse_m = (np.nanmean((pre_image[matched_building_mask] - gt_image[matched_building_mask]) ** 2)) ** 0.5
        else:
            rmse_m = 0.0


        low_rise_building_mask = (gt_image >= 1) & (gt_image < 15)
        mid_rise_building_mask = (gt_image >= 15) & (gt_image < 40)
        high_rise_building_mask = gt_image >= 40
        
        low_rise = gt_image[low_rise_building_mask]
        mid_rise = gt_image[mid_rise_building_mask]
        high_rise = gt_image[high_rise_building_mask]

        low_rise_pred = pre_image[low_rise_building_mask]
        mid_rise_pred = pre_image[mid_rise_building_mask]
        high_rise_pred = pre_image[high_rise_building_mask]

        if high_rise.size > 0 and high_rise_pred.size > 0:
            high_rise_mse = np.nanmean((high_rise - high_rise_pred) ** 2)
            high_rise_rmse = np.sqrt(high_rise_mse)
            self.total_high_rise_rmse += high_rise_rmse
            self.count_high_rise += 1  
        else:
            high_rise_rmse = None

        if mid_rise.size > 0 and mid_rise_pred.size > 0:
            mid_rise_mse = np.nanmean((mid_rise - mid_rise_pred) ** 2)
            mid_rise_rmse = np.sqrt(mid_rise_mse)
            self.total_mid_rise_rmse += mid_rise_rmse
            self.count_mid_rise += 1  
        else:
            mid_rise_rmse = None
        
        if low_rise.size > 0 and low_rise_pred.size > 0:
            low_rise_mse = np.nanmean((low_rise - low_rise_pred) ** 2)
            low_rise_rmse = np.sqrt(low_rise_mse)
            self.total_low_rise_rmse += low_rise_rmse
            self.count_low_rise += 1
        else:
            low_rise_rmse = None

        
        self.mse += mse_i
        self.rmse += rmse_i
        self.rmse_building += rmse_b
        self.rmse_matched += rmse_m
        self.abs += abs_i

        # DELTA METRICS
        delta_pre_image[delta_pre_image == 0] = 0.00001
        delta_pre_image[delta_pre_image < 0] = 999
        delta_gt_image[delta_gt_image <= 0] = 0.00001
        maxRatio = np.maximum(delta_pre_image / delta_gt_image, delta_gt_image / delta_pre_image)
        self.delta1_sum += (maxRatio < 1.25).mean()
        self.delta2_sum += (maxRatio < 1.25 ** 2).mean()
        self.delta3_sum += (maxRatio < 1.25 ** 3).mean()
        
        self.img_sample += 1

    def calculate_metrics(self):
        #mse = np.nanmean(self.mse_list)
       
        #mae = np.nanmean(self.abs_list)
        #rmse = np.nanmean(self.rmse_list)
        #rmse_building = np.nanmean(self.rmse_building_list)
        
        #delta1 = np.nanmean(self.delta1)#self.delta1 / self.total_samples
        #delta2 = np.nanmean(self.delta2)#self.delta2 / self.total_samples
        #delta3 = np.nanmean(self.delta3)#self.delta3 / self.total_samples
        mse = self.mse / self.img_sample
        rmse = self.rmse / self.img_sample
        rmse_building = self.rmse_building / self.img_sample
        rmse_matched = self.rmse_matched / self.img_sample
        high_rise_rmse = self.total_high_rise_rmse / self.count_high_rise if self.count_high_rise > 0 else 0
        mid_rise_rmse = self.total_mid_rise_rmse / self.count_mid_rise if self.count_mid_rise > 0 else 0
        low_rise_rmse = self.total_low_rise_rmse / self.count_low_rise if self.count_low_rise > 0 else 0
        mae = self.abs / self.img_sample
        delta1 = self.delta1_sum / self.img_sample
        delta2 = self.delta2_sum / self.img_sample
        delta3 = self.delta3_sum / self.img_sample
        
        return mse, rmse, rmse_building, rmse_matched, high_rise_rmse, mid_rise_rmse, low_rise_rmse, mae, delta1, delta2, delta3