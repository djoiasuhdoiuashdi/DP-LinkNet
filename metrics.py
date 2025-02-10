

import argparse
import cv2
import numpy as np
import math 
from scipy.signal import fftconvolve, convolve
from scipy.ndimage import distance_transform_edt
from numba import njit
from decimal import Decimal, ROUND_UP, getcontext

# 100% Correct
def create_weight_matrix(mask_size=5):
    weight_matrix = np.zeros((mask_size, mask_size), dtype=np.float64)
    center = mask_size//2

    for i in range(mask_size):
        for j in range(mask_size):
            if i == center and j == center:
                weight_matrix[i, j] = 0.0  # Center weight is zero
            else:
                distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                weight_matrix[i, j] = 1.0 / distance
  
    # Normalize the weight matrix
    sum_weights = np.sum(weight_matrix)

    return weight_matrix / sum_weights

@njit
def DRDcalc(i, j, image_gt, image, normalized_weight_matrix, mask_size):
   
    #Dimensions
    height, width = image_gt.shape

    #Value of the k-th fliiped pixel
    # at the processed image g
    value_gk=image[i,j]

    #Construct mxm Block Bk of gt image (f)
    #and measure the Difference matrix Dk
    Bk = np.zeros((mask_size, mask_size), dtype=np.float64)
    Dk = np.zeros((mask_size, mask_size), dtype=np.float64)
    h = 2
    for x in range(mask_size):
        for y in range(mask_size):
            if(i-h+x < 0 or j-h+y < 0 or i-h+x >= height or j-h+y >= width):
                Bk[x,y] = value_gk
            else:
                Bk[x,y] = image_gt[i-h+x,j-h+y]; 
            Dk[x,y] = abs(Bk[x,y]-value_gk); 
        
   

    #The distortion DRDk for the k-th pixel
        #of the processed image g is "ret"
        DRDk = Dk * normalized_weight_matrix

    res = np.sum(DRDk)
    # file.write(f"{i} {j} {res}\n")

    return res



# 100% Correct
def NUBNcalc(f, ii, jj, blck):
    """
    Calculate NUBN for a specific block in the array.

    Parameters:
    - f (2D array): The input two-dimensional array.
    - ii (int): Block index along the x-axis (1-based).
    - jj (int): Block index along the y-axis (1-based).
    - blck (int): Size of each block.

    Returns:
    - retb (int): 1 if there's a change in the block, else 0.
    """
    # Convert to zero-based indices for Python
    startx = (ii - 1) * blck
    endx = ii * blck
    starty = (jj - 1) * blck
    endy = jj * blck

    check_prv = -2
    retb = Decimal("0")

    for xx in range(startx, endx):
        for yy in range(starty, endy):
            check = f[xx, yy]
            if check_prv < 0:
                check_prv = check
            else:
                if check != check_prv:
                    retb = 1
                    break  # Exit the inner loop
        if retb != 0:
            break  # Exit the outer loop if a change is detected

    return retb

def get_drd(im, im_gt, normalized_weight_matrix):

    getcontext().prec = 20
    im = im.astype(np.float64)
    im_gt = im_gt.astype(np.float64)
    img_height, img_width = im.shape  
    
    block_size=8 
    n = 2
    mask_size = 2 * n + 1  

    # Calculate NUBN
    print("*"*20)
    print("Calculating NUBN")
    total_nubn = Decimal("0.0")
    xb =  img_height// block_size
    yb = img_width // block_size   

    # Assuming xb and yb are defined, and z is a 2D list or array
    for i in range(1, xb + 1):
        for j in range(1, yb + 1):
            nubn_b = NUBNcalc(im_gt, i, j, block_size)
            total_nubn += nubn_b
            

    print("*"*20)
    print("NUBN is:", total_nubn)
   
    print("*"*20)
    print("Calculating total_drd")

    np.set_printoptions(precision=20, suppress=True, threshold=np.inf)
    total_drd = Decimal("0.0")
    
    print(normalized_weight_matrix)
    
    #--------------------------
    # Official Approach
    #--------------------------
  
    difference_image = np.where(np.abs(im - im_gt) > 0.5)
    # with open('numbers.txt', 'w') as file:   
    for i, j in zip(difference_image[0], difference_image[1]):
        res = DRDcalc(i, j, im_gt, im, normalized_weight_matrix, mask_size)
        decimal_value = Decimal(str(res))

        # Limit to 6 digits after the decimal point
        decimal_value = decimal_value.quantize(Decimal('0.0000001'))
        total_drd += decimal_value
        # file.write(f"{i+2} {j+2} {decimal_value}\n")
            
   
    #--------------------------
    # My old approach / didnt confirm to specification
    #--------------------------

    # flipped = (im != im_gt).astype(np.float64)
    # total_drd_map = fftconvolve(flipped, normalized_weight_matrix, mode="same")
    # total_drd = Decimal(total_drd_map[flipped == 1].sum())
  
    #total_drd = 3159824.24879971519112586975
    
    #-----------------------------------

    print("*"*20)
    print("Total DRD is:", total_drd)

    return total_drd / total_nubn



def load_image_as_binary(image_path):
    """Load a black and white image and convert it to a binary array."""
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Convert the image to binary (0 and 1)
    binary_array = (image > 0).astype(np.uint8)
    return binary_array

# 100% correct
def calculate_metrics(im, im_gt, r_weight, p_weight):
    
    height, width = im_gt.shape
    
    # Compute DRD metric
    normalized_weight_matrix = create_weight_matrix()
    drd = get_drd(im, im_gt, normalized_weight_matrix) 
    
    # Create masks
    TP_mask = (im == 0) & (im_gt == 0)
    FP_mask = (im == 0) & (im_gt == 1)
    FN_mask = (im == 1) & (im_gt == 0)
    
    # Compute weighted weights
    weighted_p_weight = 1.0 + p_weight
    weighted_r_weight = 1.0 + r_weight
    
    # Sum weighted weights using masks
    TPwp = weighted_p_weight[TP_mask].sum()
    FPwp = weighted_p_weight[FP_mask].sum()
    TPwr = r_weight[TP_mask].sum()
    FNwr = r_weight[FN_mask].sum()
    
    # Compute weighted precision, recall, and F-measure
    w_precision = TPwp / (TPwp + FPwp) if (TPwp + FPwp) > 0 else 0.0
    w_recall = TPwr / (TPwr + FNwr) if (TPwr + FNwr) > 0 else 0.0
    w_f_measure = (2 * w_precision * w_recall) / (w_precision + w_recall) if (w_precision + w_recall) > 0 else 0.0
    
    # Compute standard precision, recall, and F-measure
    TP = TP_mask.sum()
    FP = FP_mask.sum()
    FN = FN_mask.sum()
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Compute MSE and PSNR
    npixel = height * width
    mse = (FP + FN) / npixel
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
    # Print metrics with formatted output
    
    return f_measure, w_f_measure, psnr, drd, recall, precision, w_recall, w_precision

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare two binarized images with a weighting factor.')
    parser.add_argument('input_image', type=str, help='Path to the predicted binarized image.')
    parser.add_argument('gt_image', type=str, help='Path to the ground truth binarized image.')
    parser.add_argument('r_weight', type=str, help='Path to the r-weight for the ground truth image.')
    parser.add_argument('p_weight', type=str, help='Path to the p-weight for the ground truth image.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    # Load images
    im = load_image_as_binary(args.input_image)
    im_gt = load_image_as_binary(args.gt_image)
    
    # Load weights
    height, width = im_gt.shape
    r_weight = np.loadtxt(args.r_weight, dtype=np.float64).flatten()[:height * width].reshape((height, width))
    p_weight = np.loadtxt(args.p_weight, dtype=np.float64).flatten()[:height * width].reshape((height, width))
    
    f_measure, w_f_measure, psnr, drd, recall, precision, w_recall, w_precision = calculate_metrics(im, im_gt, r_weight, p_weight)
    
    print(f"F-Measure: {f_measure * 100:.4f}")
    print(f"Pseudo F-Measure (Fps): {w_f_measure * 100:.4f}")
    print(f"PSNR: {psnr:.4f}")
    print(f"DRD: {drd:.4f}")
    print(f"Recall: {recall * 100:.4f}")
    print(f"Precision: {precision * 100:.4f}")
    print(f"Pseudo-Recall (Rps): {w_recall * 100:.4f}")
    print(f"Pseudo-Precision (Pps): {w_precision * 100:.4f}")
    

    

    
    

