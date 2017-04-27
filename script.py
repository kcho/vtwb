import numpy as np
import nibabel as nb
import nilearn
import scipy

def corr2_coeff(A,B):
    # http://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

def main():
    # load nifti file
    #thal = nb.load('/Users/kangik/Downloads/atlasHO_bilateral_thalamus_ds.nii.gz')
    thal = nb.load('/Users/kangik/Downloads/B_thal_on_filtered_func_data_ds.nii.gz')
    wb = nb.load('/Users/kangik/Downloads/filtered_func_data_ds.nii.gz')

    # read matrix from the nifti file
    thald = thal.get_data()
    wbd = wb.get_data()
    print(wbd.shape)

    volume_shape = wbd.shape[:-1]
    coords = list(np.ndindex(volume_shape))

    thal_x, thal_y, thal_z = np.where(thald[:,:,:,0] != 0)
    thal_coords = np.array((thal_x, thal_y, thal_z)).T

    thald_nonzero = thald[np.where(thald[:,:,:,0] != 0)]

    wbd_reshape = wbd.reshape(len(coords), wbd.shape[3])
    thald_reshape = thald_nonzero.reshape(len(thal_coords), wbd.shape[3])

    print(thald_reshape.shape)

    corrMap = corr2_coeff(wbd_reshape, thald_reshape)
    corrMap_reshape = corrMap.reshape(volume_shape[0], volume_shape[1], volume_shape[2], len(thal_coords))

    ## save image
    img = nb.Nifti1Image(corrMap_reshape, affine=thal.affine)
    img.to_filename('prac.nii.gz')


    # enumerate over thalamus ts
    #thal_voxel_corr_map_list = []

    #for (x,y,z), thal_val in np.ndenumerate(thald[:,:,:,0]):
        ## make empty 3d image
        #thal_voxel_corr_map = np.zeros_like(thald[:,:,:,0])

        ## for thalamus ts voxels
        #if thal_val != 0:
            #thal_ts = thald[x,y,z,:]
            #for (bx, by, bz), brain_val in np.ndenumerate(wbd[:,:,:,0]): 
                #if brain_val != 0: 
                    #brain_ts = wbd[bx,by,bz,:]
                    ### correlation 
                    #coeff, pval = scipy.stats.pearson(thal_ts, brain_ts)

                    ## save the coefficient at the voxel location
                    #thal_voxel_corr_map[bx,by,bz] = coeff

        ## append the 3d map to the list
        #thal_voxel_corr_map_list.append(thal_voxel_corr_map)

    ## merge list of 3d maps into 4d image
    #four_d_map = np.concatenate([x[..., np.newaxis] for x in thal_voxel_corr_map_list], axis=3)

    ## save image
    #img = nb.Nifti1Image(four_d_map, affine=thal.affine)
    #img.to_filename('prac.nii.gz')
if __name__ == '__main__':
    main()
