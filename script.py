import numpy as np
import nibabel as nb
import scipy
from scipy.stats import pearsonr

def main():
    print('start')
    # load nifti file
    thal = nb.load('/Users/kangik/Downloads/B_thal_on_filtered_func_data_ds.nii.gz')
    wb = nb.load('/Users/kangik/Downloads/filtered_func_data_ds.nii.gz')

    # read matrix from the nifti file
    thald = thal.get_data()
    wbd = wb.get_data()

    # enumerate over thalamus ts
    thal_voxel_corr_map_list = []

    for (x,y,z), thal_val in np.ndenumerate(thald[:,:,:,0]):
        # make empty 3d image
        thal_voxel_corr_map = np.zeros_like(thald[:,:,:,0])

        # for thalamus ts voxels
        if thal_val != 0:
            print(x,y,z)
            thal_ts = thald[x,y,z,:]
            for (bx, by, bz), brain_val in np.ndenumerate(wbd[:,:,:,0]): 
                if brain_val != 0: 
                    brain_ts = wbd[bx,by,bz,:]
                    ## correlation 
                    coeff, pval = pearsonr(thal_ts, brain_ts)

                    # save the coefficient at the voxel location
                    thal_voxel_corr_map[bx,by,bz] = coeff

            # append the 3d map to the list
            thal_voxel_corr_map_list.append(thal_voxel_corr_map)

    # merge list of 3d maps into 4d image
    four_d_map = np.concatenate([x[..., np.newaxis] for x in thal_voxel_corr_map_list], axis=3)

    # save image
    img = nb.Nifti1Image(four_d_map, affine=thal.affine)
    img.to_filename('prac.nii.gz')
if __name__ == '__main__':
    main()
