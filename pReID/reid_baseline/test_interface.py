
from interface import Reid_baseline
# import interface.Reid_baseline 


if __name__ == "__main__":

    query_path   = '/home/luigy/luigy/develop/re3/tracking/dataset_prid2011/replicate_FFPRID_track/A-B/A-B_000001/otherCam_person_0017.png'
    # query_path   = '/home/luigy/luigy/develop/re3/tracking/dataset_prid2011/replicate_FFPRID_track/A-B/A-B_000001/thisCam_person_0017.png'
    # gallery_path = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/testing/pack_v0_top_10_numInlier_1/gallery_inliers'
    gallery_path = '/home/luigy/luigy/develop/re3/tracking/dataset_prid2011/replicate_FFPRID_track/A-B/A-B_000001/outcome_otherCam_person_0017_all_Test'

    save_path    = '/home/luigy/luigy/develop/re3/tracking/dataset_prid2011/replicate_FFPRID_track/A-B/A-B_000001/out_test'

    top_k        = 20
 
    model                = Reid_baseline()
    imgs_path, dist_mat  = model.predict(query_path, gallery_path)
    # breakpoint()
    model.save_patch_results(query_path, imgs_path, dist_mat, save_path,  top_k=top_k, threshold=1.0)

