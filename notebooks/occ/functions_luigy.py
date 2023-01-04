

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from PIL import Image
# dataset
import os
import glob
import math
from matplotlib.colors import ListedColormap  


import pickle
import gc

from shutil import copy2


shape_img = (224,224,3)




def get_array_distances(X_target, X_matrix):
    result=[]
    for i in X_matrix:
        dist = np.linalg.norm(X_target - i , axis=1)
        result.append(dist)
    return np.asarray(result)


def get_array_for_sorted( array_predicts, out_mean=None ):
    num_images = len(array_predicts)
    if out_mean is not None:
        X_mean = out_mean
    else:
        X_mean = array_predicts.mean(axis=0)[None,:]
        
    array_distances = get_array_distances(X_mean, array_predicts)
    indices_sorted  = np.argsort(array_distances, axis = 0)
    return array_distances[indices_sorted], indices_sorted, X_mean


def sorted_predicts(predict_inliers):
    if predict_inliers.size==0:
        mean_arr = 0.0
    else:
        mean_arr = predict_inliers.mean(axis=0)[None,:]
    
    array_distances_sorted, indices_sorted, mean = get_array_for_sorted(predict_inliers, mean_arr)
    return array_distances_sorted, indices_sorted, mean

def apply_query_ON_array(query, *arrays):
    results = []
    for arr in arrays:
        if shape_img==arr.shape[-3:]:
            selection = arr[query]
            selection = selection.reshape(-1, *selection.shape[-3:])
            results.append(selection)
        else:
            selection = arr[query]
            selection = selection.reshape(-1)
            results.append(selection)
    return tuple(results)


def copy_file(src, dst): 
    return copy2(src, dst)


def save_model(obj, name=None,  ):
    abs_path = None
    if name is None:
        name = "temporal_name.pkl"
    abs_path = name
    pickle.dump(obj, open(abs_path, 'wb'))
    return True
 
def load_model(path):
    return pickle.load(open(path , 'rb'))        



def create_dir(*path):
    list_paths = list()
    for i in path:
        list_paths.append(_create_dir(i))
    if len(list_paths)>1:
        return tuple(list_paths)
    elif len(list_paths)==1:
        return list_paths[0]
    else:
        return False


def _create_dir(path):
    try:
        os.makedirs( path ,exist_ok=True)
        return path
    except:
        print("{} already exist".format(path))
        return None



def next_power_of_two(number):
    i = 2
    while i**2<number:
        i+=1
    return i
    

def add_sth_path(full_path, sth, dst_path=None, prefix=None):
    assert os.path.isfile(full_path), "err, don't exist file: {}".format(full_path)
    
    _path, _name = os.path.split(full_path)
    if dst_path is not None:
        _path = dst_path
    _prefix     = 'xxx'
    if prefix is not None:
        _prefix = prefix
    if type(sth)==int: # "sth is a number"
        _new_name = '{}_{:08d}_{}'.format(_prefix, sth, _name)
    else:
        _new_name = '{}_{}_{}'.format(_prefix, sth, _name)
    return os.path.join(_path, _new_name)


def openDir_images(paths, shape=None, normalize=True ):
    list_out = []
    for path in paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if shape is not None:
        	img = cv2.resize(img, (shape[0], shape[1]))
        if normalize:
            img = img.astype('float32')/255
        list_out.append(img)
    return np.array(list_out)


def open_mydata(data_path, output_shape=(224,224)):

    #typedata : numpy array
    #typelabel : [0 0 0 1] #normal_cp #abnormal_cp #normal_smd #abnormal_smd
    x_train_normal, x_test_normal, x_test_abnormal = [], [], []
    y_train_normal = []
    x_test_normal_path , x_test_abnormal_path = [] ,[]
    cp1_path = os.path.join(data_path, 'one_class_classification')

    #make reference data
    cp1_normal_path = os.path.join(cp1_path, 'train', 'person')
    cp1_normal = sorted(glob.glob('{}/*'.format(cp1_normal_path)))
    for cp1 in cp1_normal:
        x_train_normal.append(cp1)
        y_train_normal.append(np.array([1, 0]))
        
    
    # #make test data
    cp1_test_path     = os.path.join(cp1_path, 'test')
    cp1_test_normal   = os.path.join(cp1_test_path,'person')
    cp1_test_abnormal = os.path.join(cp1_test_path,'no_person')

    
    cp1_test_normal_files = sorted(glob.glob('{}/*'.format(cp1_test_normal)))
    for cp1_test_norfile in cp1_test_normal_files : 
        x_test_normal_path.append(cp1_test_norfile)
        x_test_normal.append(cp1_test_norfile)

    cp1_test_abnormal_files = sorted(glob.glob('{}/*'.format(cp1_test_abnormal)))
    for cp1_test_abnor in cp1_test_abnormal_files:
        x_test_abnormal_path.append(cp1_test_abnor)
        x_test_abnormal.append(cp1_test_abnor)
    
    #resize data
    # x_train_normal = resize_data(x_train_normal)
    x_train_normal = openDir_images(x_train_normal, output_shape)
    y_train_normal = np.array(y_train_normal)

    x_test_normal = openDir_images(x_test_normal, output_shape)
    x_test_abnormal = openDir_images(x_test_abnormal, output_shape)
    # x_train_normal : normal data
    # x_test_normal : test normal data
    # x_test_abnormal : test abnormal data
    
    return x_train_normal, y_train_normal, x_test_normal, x_test_abnormal, x_test_normal_path, x_test_abnormal_path



def openImages_FromDirectory(path, keyid=None, shape=None, normalize= True):

    # path_images = sorted(glob.glob('{}/*.{}'.format(base_path, extension)))
    if keyid is None:
        base_path   = path
    else:
        base_path   = os.path.join(path, keyid)
    path_images = sorted( os.listdir(base_path) )
    path_images = [os.path.join(base_path,i) for i in path_images]
    images      = openDir_images(path_images, shape, normalize=normalize )
    return images, path_images


### plot images

def set_blank_box(axs, blank_img):
    ax = axs.ravel()
    for ax_i in ax:
        ttt = ax_i.imshow(blank_img)
    for ax_i in ax:
        ax_i.set(xticks=[], yticks=[])
    return ax


def get_dim(images_array, size_plot):
    n_images  = images_array.shape[0]
    nrows     = min( size_plot, int(math.ceil(n_images/size_plot)) )
    ncols     = max( size_plot, int(math.ceil(n_images/size_plot)) )
    return n_images, nrows, ncols


def set_grid(images_array):
    blank_img = np.full_like(images_array[0], 255, dtype=np.uint8)
    return blank_img



def get_color(n_images, err, color_top=None, color_bottom=None, name_color='other', inlier=True):
    if n_images==1:
        if inlier:
            return ListedColormap(['red'])
        else:
            return ListedColormap(['black'])

    if color_top is None and color_bottom is None :
#         color_top    = 'Oranges_r'
        color_top    = 'autumn'
        color_bottom = 'Blues'
        name_color   = 'OrangeBlue'

    top       = plt.get_cmap(color_top,    int(n_images))
    bottom    = plt.get_cmap(color_bottom, int(n_images))

    part_top      = int(n_images/2)-err
    part_bottom   = int(n_images/2)+err
    top_select    = np.linspace(0.0, 0.5, part_top)
    bottom_select = np.linspace(0.5, 1.0, part_bottom)
    newcolors     = np.vstack(( top(top_select),
                                bottom(bottom_select)
                              ))
    color_map          = ListedColormap(newcolors, name=name_color) 
    
    return color_map







def show_imagesGrid(images_array, 
                    labels,
                    array_distances = None, 
                    classes_dict    = None, 
                    
                    images_array_outliers    = None, 
                    labels_outliers          = None,
                    classes_dict_outliers    = None,
                    
                    savePath        = None, 
                    saveName        = None, 
                    figsize         = (14,18),
                    key             = None,
                    noisy           = False,
                    imshow          = True,
                    caption_msg     = None,
                    ):
    
    flag_zero_imgs_in  = True
    flag_zero_imgs_out = True

    num_imgs_default = 20
    if images_array.size==0:
        images_array = np.zeros((num_imgs_default, 224, 224, 3), dtype='uint8')
        flag_zero_imgs_in = False

    if labels.size==0:
        labels = np.arange(num_imgs_default, dtype='uint8')
        
#     assert images_array.ndim == 4 and labels.ndim == 1, 'error {}'.format("images_array.ndim == 4 and labels.ndim == 1")
#     if  len(images_array) == 0:
#         flag_zero_imgs_in = False
        
    if images_array_outliers is None:
        flag_zero_imgs_out = False
    elif len(images_array_outliers)==0:
        flag_zero_imgs_out = False
    
    tmp = None
    err = 0
    
    if imshow is False:
        plt.ioff()

    ############################################################################################ 
    if images_array_outliers is not None:
        _size_plot = max(next_power_of_two(len(images_array)),next_power_of_two(len(images_array_outliers)))
    else:
        _size_plot = next_power_of_two(len(images_array))
    
    size_plot = max(_size_plot,5)
    
    n_images, nrows, ncols = get_dim(images_array, size_plot)
    blank_img              = set_grid(images_array)
    color_map              = get_color(n_images, err )

    if flag_zero_imgs_out:
        n_images_out, nrows_out, ncols_out = get_dim(images_array_outliers, size_plot)
        blank_img_out                      = set_grid(images_array_outliers)
        color_map_outliers                 = get_color(n_images_out, err, color_top='bone', color_bottom='bone_r', inlier=False)
    else:
        n_images_out, nrows_out, ncols_out = 0,0,0
        blank_img_out                      = 0
        color_map_outliers                 = 0
    
   
    ############################################################################################    
    height_ratios = [1, 1]
    width_ratios  = [1, 0.025]

    fig        = plt.figure(figsize=figsize, constrained_layout=False, dpi=200)
    outer_grid = fig.add_gridspec(2, 2, wspace=0.1, hspace=0.2, height_ratios=height_ratios, width_ratios=width_ratios)

    for ax_tmp in fig.get_axes():
        ax_tmp.set(xticks=[], yticks=[])

    ############################################################################################
    fig.text(.5, .05, caption_msg, ha='center', size=12) #, weight='bold')

    figtext_kwargs = dict(horizontalalignment ="center",  
                          fontsize = 14, color ="black",  
                          style ='normal', wrap = True,
                          fontweight = 'black', size='xx-large' ) 
    if flag_zero_imgs_in:
        title_inliers  = '{} Inliers '.format(n_images)
    else:
        title_inliers  = '{} Inliers '.format(0)
    title_outliers = '{} Outliers'.format(n_images_out)
    t1        = plt.figtext(0.5, 0.92, title_inliers,  **figtext_kwargs)
    t2        = plt.figtext(0.5, 0.5 , title_outliers, **figtext_kwargs)
    
    ##################### *****
    nrows     = max(nrows,nrows_out)
    nrows_out = max(nrows,nrows_out)
    
    ncols     = max(ncols,ncols_out)
    ncols_out = max(ncols,ncols_out)
    ##################### 0,0    
    width_ratios  =  [1 for _ in range(ncols)] 
    height_ratios =  [1 for _ in range(nrows)]
    block0_inner_grid = outer_grid[0, 0].subgridspec(nrows=nrows, ncols=ncols, hspace=0.6 , wspace=0.1, height_ratios=height_ratios, width_ratios=width_ratios)  
    axs1              = block0_inner_grid.subplots()  
    ##################### 1,0
    width_ratios_out  =  [1 for _ in range(ncols_out)]
    height_ratios_out =  [1 for _ in range(nrows_out)]
    block1_inner_grid = outer_grid[1, 0].subgridspec(nrows=nrows_out, ncols=ncols_out, hspace=0.6 , wspace=0.1, height_ratios=height_ratios_out, width_ratios=width_ratios_out)  
    axs2              = block1_inner_grid.subplots() 
    ##################### 0,1 COLORBAR
    block3_inner_grid = outer_grid[0, 1].subgridspec(nrows=1, ncols=1)
    axs3              = block3_inner_grid.subplots()  # Create all subplots for the inner grid.
    ############################################################################################
    
    if array_distances is not None :
        tmp = array_distances.copy() 
        tmp = tmp.reshape(len(array_distances))
    if noisy:
        images_array = get_noisy_data(images_array,_sigma=0.155)
    ax1 = set_blank_box(axs1, blank_img)
    ax2 = set_blank_box(axs2, blank_img)
    
    ####################################
    if flag_zero_imgs_in:
        for i in range(n_images):
           # Set the borders to a given color...
            ax1[i].yaxis.label.set_color(color_map(i))
            ax1[i].xaxis.label.set_color(color_map(i))
            for spine in ax1[i].spines.values():
                spine.set_edgecolor(color_map(i))
                spine.set_linewidth(5)

            if classes_dict:
                title = classes_dict[labels[i]]
            elif array_distances is not None:
                title = "ID_{}:    {:.3f}".format(labels[i],tmp[i])
            else:
                title = "ID_{}".format(int(labels[i]))

            ax1[i].set_title(str(title), fontweight="bold")
            ax1[i].title.set_color(color_map(i))

            ttt = ax1[i].imshow(images_array[i])        
        for ax_tmp in ax1[n_images:]:
            ax_tmp.axis('off')
    else:
        for ax_tmp in ax1:
            ax_tmp.axis('off')
    
    
    ####################################
    if  images_array_outliers is not None:
        n_images_outliers =  images_array_outliers.shape[0]

        for i in range(n_images_outliers):
            ax2[i].yaxis.label.set_color(color_map_outliers(i))
            ax2[i].xaxis.label.set_color(color_map_outliers(i))
            for spine in ax2[i].spines.values():
                spine.set_edgecolor(color_map_outliers(i))
                spine.set_linewidth(5)

            if classes_dict:
                title_outliers = classes_dict_outliers[labels_outliers[i]]
            else:
                title_outliers = "ID_{}".format(int(labels_outliers[i]))

            ax2[i].set_title(str(title_outliers), fontweight="bold")
            ax2[i].title.set_color(color_map_outliers(i))
            ttt2 = ax2[i].imshow(images_array_outliers[i])     
                
        for ax_tmp in ax2[n_images_outliers:]:
            ax_tmp.axis('off')

    ####################################

    # Normalizer
    list_ticks = None
    norm = None
    
    if flag_zero_imgs_in is False:
        labels     = [labels[0]]
        list_ticks = np.linspace(min(labels),max(labels),n_images).tolist()        
        norm       = mpl.colors.Normalize(vmin=min(labels), vmax=max(labels) )
    
    elif array_distances is not None:
        list_ticks = np.linspace(min(tmp),max(tmp),n_images).tolist()
        norm       = mpl.colors.Normalize(vmin=min(array_distances), vmax=max(array_distances)) 
        
    else:
        list_ticks = np.linspace(min(labels),max(labels),n_images).tolist()        
        norm       = mpl.colors.Normalize(vmin=min(labels), vmax=max(labels) ) 

    # creating ScalarMappable 
    sm = plt.cm.ScalarMappable(cmap=color_map.reversed(), norm=norm)
    sm.set_array([])
    

    
    labels_temp = None
    if array_distances is not None and flag_zero_imgs_in:
        labels_temp     = ["{:.3f} ".format(i) for i in np.flip(tmp)]
        labels_temp[-1] = labels_temp[-1] + " >> BEST  \n            CANDIDATE"
        labels_temp[0]  = labels_temp[0]  + " >> WORST \n            CANDIDATE"
    else:
        labels_temp     = ["{}".format(i) for i in np.flip(labels)]
        labels_temp[0]  = labels_temp[0]  + ">>   LABEL"
        labels_temp[-1] = labels_temp[-1] + ">>   LABEL"
    
    cbar = fig.colorbar(sm, cax=axs3, ticks=list_ticks ) #, extend='both',shrink=0.9,)
    
    if key is not None:
        cbar.set_label(label='KEY ID: {}'.format(key), size=15, weight='bold')
    
    if n_images > 1 and flag_zero_imgs_in:
        cbar.ax.tick_params(labelsize=10,length=8, width=8)
        cbar.ax.set_yticklabels(labels_temp, weight='bold')  # vertically oriented colorbar


    if savePath is not None:
        assert os.path.exists(savePath), 'savePath: {} doesnt exist'.format(savePath) 
        if saveName is None:
            saveName = 'testGrid.png'
#         fig.savefig(os.path.join(savePath, saveName), dpi=fig.dpi)
        fig.savefig(os.path.join(savePath, saveName), dpi=200 , bbox_inches='tight') 
    
    if imshow is False:
        plt.close(fig)

        # fig.clf()
        # plt.close()
        # gc.collect()
        plt.figure().clear()
        plt.close()






# def show_imagesGrid_v0(images_array, 
#                     labels, 
#                     array_distances = None, 
#                     classes_dict    = None, 
#                     savePath        = None, 
#                     saveName        = None, 
#                     figsize         = (14,16),
#                     key             = None,
#                     noisy           = False,
#                     imshow          = True,
#                     caption_msg     = None,
# ):
    
#     assert images_array.ndim == 4 and labels.ndim == 1, 'error {}'.format("images_array.ndim == 4 and labels.ndim == 1")
    
#     if imshow is False:
#         plt.ioff()

#     blank_img = np.full_like(images_array[0], 255, dtype=np.uint8)

#     n_images  = images_array.shape[0]
#     size_plot = next_power_of_two(n_images)
#     size_plot = 4 if size_plot==1 else size_plot
#     nrows     = min( size_plot, int(math.ceil(n_images/size_plot)) ) 
#     ncols     = size_plot
#     figsize   = (figsize[0], min(int(nrows*(figsize[1]/size_plot)),figsize[1]))

#     top       = plt.get_cmap('Oranges_r', int(n_images))
#     bottom    = plt.get_cmap('Blues', int(n_images))

#     tmp = None
#     err = 0
    
#     part_top      = int(n_images/2)-err
#     part_bottom   = int(n_images/2)+err
#     top_select    = np.linspace(0.0, 0.5, part_top)
#     bottom_select = np.linspace(0.5, 1.0, part_bottom)
#     newcolors = np.vstack((top(top_select),
#                            bottom(bottom_select)
#                           ))
#     color_map = ListedColormap(newcolors, name='OrangeBlue')
#     width_ratios  =  [1 for _ in range(ncols)]
#     height_ratios =  [1 for _ in range(nrows)]
    
#     fig, ax   = None, None
#     fig, ax   = plt.subplots(   nrows,
#                                 ncols, 
#                                 figsize = figsize, 
#                                 subplot_kw={'xticks':(), 'yticks': ()},
#                                 gridspec_kw={'width_ratios': width_ratios,'height_ratios':height_ratios}
#                               )
#     fig.text(.5, .05, caption_msg, ha='center', size=16) #, weight='bold')
    
#     if array_distances is not None :
#         tmp = array_distances.copy() 
#         tmp = tmp.reshape(len(array_distances))
        
#     if noisy:
#         images_array = get_noisy_data(images_array,_sigma=0.155)
    
#     ax = ax.ravel()
#     for ax_i in ax:
#         ttt = ax_i.imshow(blank_img)        

    
#     for i in range(n_images):
#     #     pixels = images[i].reshape(-1,28)
#     #     pixels = cv2.resize(images[i],(224,224),interpolation=cv2.INTER_AREA)
#     #     ax[i].imshow(pixels)

#        # Set the borders to a given color...
#         ax[i].yaxis.label.set_color(color_map(i))
#         ax[i].xaxis.label.set_color(color_map(i))
#         for spine in ax[i].spines.values():
#             spine.set_edgecolor(color_map(i))
#             spine.set_linewidth(5)

#         if classes_dict:
#             title = classes_dict[labels[i]]
#         elif array_distances is not None:
# #             title = '{:.3f}'.format(tmp[i])
#             title = "ID {}:  {:.3f}".format(labels[i],tmp[i])
#         else:
#             title = int(labels[i])
#         ax[i].set_title(str(title), fontweight="bold")
#         ax[i].title.set_color(color_map(i))
        
#         ttt = ax[i].imshow(images_array[i])        
# #         ttt = ax[i].plot(images_array[i])        


#     # Normalizer
#     list_ticks = None
#     norm = None
#     if array_distances is not None:
#         list_ticks = np.linspace(min(tmp),max(tmp),n_images).tolist()
#         norm       = mpl.colors.Normalize(vmin=min(array_distances), vmax=max(array_distances)) 
#     else:
#         list_ticks = np.linspace(min(labels),max(labels),n_images).tolist()        
#         norm       = mpl.colors.Normalize(vmin=min(labels), vmax=max(labels) ) 

#     # creating ScalarMappable 
#     sm = plt.cm.ScalarMappable(cmap=color_map.reversed(), norm=norm)
#     sm.set_array([])
    
#     cbar = fig.colorbar(sm, ax=ax.tolist(), ticks=list_ticks ) #, extend='both',shrink=0.9,)
# #     cbar = fig.colorbar(sm, ax=ax.tolist()) #, extend='both',shrink=0.9,)
#     if key is not None:
#         cbar.set_label(label='KEY ID: {}'.format(key), size=15, weight='bold')

#     cbar.ax.tick_params(labelsize=10,length=8, width=8)


    
#     labels_temp = None
#     if array_distances is not None:
#         labels_temp     = ["{:.3f} ".format(i) for i in np.flip(tmp)]
# #         labels_temp     = ["{:.3f} ".format(i) for i in tmp]
#         labels_temp[-1] = labels_temp[-1] + " >> BEST  \n            CANDIDATE"
#         labels_temp[0]  = labels_temp[0]  + " >> WORST \n            CANDIDATE"
#     else:
#         labels_temp     = ["{}".format(i) for i in np.flip(labels)]
#         labels_temp[0]  = labels_temp[0]  + ">>   LABEL"
#         labels_temp[-1] = labels_temp[-1] + ">>   LABEL"
        
#     cbar.ax.set_yticklabels(labels_temp, weight='bold')  # vertically oriented colorbar
#     # cbar.ax.set_yticklabels(labels_temp)  # vertically oriented colorbar


#     if savePath is not None:
#         assert os.path.exists(savePath), 'savePath: {} doesnt exist'.format(savePath) 
#         if saveName is None:
#             saveName = 'testGrid.png'
# #         fig.savefig(os.path.join(savePath, saveName), dpi=fig.dpi)
#         fig.savefig(os.path.join(savePath, saveName), dpi=200) 
    
#     if imshow is False:
#         plt.close(fig)

#         # fig.clf()
#         # plt.close()
#         # gc.collect()
#         plt.figure().clear()
#         plt.close()




