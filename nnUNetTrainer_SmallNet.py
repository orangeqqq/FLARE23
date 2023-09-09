import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import *

from nnunetv2.inference.export_prediction2 import export_prediction_from_logits, resample_and_save

class nnUNetTrainer_SmallNet(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
        self.save_every = 10

    # simgle process
    # def perform_actual_validation(self, save_probabilities: bool = False):
    #     self.set_deep_supervision_enabled(False)
    #     self.network.eval()

    #     predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
    #                                 perform_everything_on_gpu=True, device=self.device, verbose=False,
    #                                 verbose_preprocessing=False, allow_tqdm=False)
    #     predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
    #                                     self.dataset_json, self.__class__.__name__,
    #                                     self.inference_allowed_mirroring_axes)

    #     # with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
    #     #     worker_list = [i for i in segmentation_export_pool._pool]
    #     if True:
    #         validation_output_folder = join(self.output_folder, 'validation')
    #         maybe_mkdir_p(validation_output_folder)

    #         # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
    #         # the validation keys across the workers.
    #         _, val_keys = self.do_split()
    #         if self.is_ddp:
    #             val_keys = val_keys[self.local_rank:: dist.get_world_size()]

    #         dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
    #                                     folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
    #                                     num_images_properties_loading_threshold=0)

    #         next_stages = self.configuration_manager.next_stage_names

    #         if next_stages is not None:
    #             _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

    #         results = []

    #         for k in dataset_val.keys():
    #             # proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
    #             #                                  allowed_num_queued=2)
    #             # while not proceed:
    #             #     sleep(0.1)
    #             #     proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
    #             #                                      allowed_num_queued=2)

    #             self.print_to_log_file(f"predicting {k}")
    #             data, seg, properties = dataset_val.load_case(k)

    #             if self.is_cascaded:
    #                 data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
    #                                                                     output_dtype=data.dtype)))
    #             with warnings.catch_warnings():
    #                 # ignore 'The given NumPy array is not writable' warning
    #                 warnings.simplefilter("ignore")
    #                 data = torch.from_numpy(data)

    #             output_filename_truncated = join(validation_output_folder, k)

    #             #debug zhu
    #             print(f"predicting {k},begin predict_sliding_window.")
    #             try:
    #                 prediction = predictor.predict_sliding_window_return_logits(data)
    #             except RuntimeError:
    #                 predictor.perform_everything_on_gpu = False
    #                 prediction = predictor.predict_sliding_window_return_logits(data)
    #                 predictor.perform_everything_on_gpu = True

    #             prediction = prediction.cpu()
    #             #debug zhu
    #             print(f"predicting {k},finish predict_sliding_window.")

    #             # this needs to go into background processes
    #             #debug zhu
    #             print(f"predicting {k},output seg nii.")
    #             # results.append(
    #             #     segmentation_export_pool.starmap_async(
    #             #         export_prediction_from_logits, (
    #             #             (prediction, properties, self.configuration_manager, self.plans_manager,
    #             #              self.dataset_json, output_filename_truncated, save_probabilities),
    #             #         )
    #             #     )
    #             # )

    #             # results.append(
    #             #     segmentation_export_pool.starmap(
    #             #         export_prediction_from_logits, (
    #             #             (prediction, properties, self.configuration_manager, self.plans_manager,
    #             #              self.dataset_json, output_filename_truncated, save_probabilities),
    #             #         )
    #             #     )
    #             # )
                
    #             # for debug purposes
    #             export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
    #                          self.dataset_json, output_filename_truncated, save_probabilities)
                            

    #             #debug zhu
    #             print(f"predicting {k},finish output seg nii.")

    #             # if needed, export the softmax prediction for the next stage
    #             if next_stages is not None:
    #                 for n in next_stages:
    #                     next_stage_config_manager = self.plans_manager.get_configuration(n)
    #                     expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
    #                                                         next_stage_config_manager.data_identifier)

    #                     try:
    #                         # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
    #                         tmp = nnUNetDataset(expected_preprocessed_folder, [k],
    #                                             num_images_properties_loading_threshold=0)
    #                         d, s, p = tmp.load_case(k)
    #                     except FileNotFoundError:
    #                         self.print_to_log_file(
    #                             f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
    #                             f"Run the preprocessing for this configuration first!")
    #                         continue

    #                     target_shape = d.shape[1:]
    #                     output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
    #                     output_file = join(output_folder, k + '.npz')

    #                      #debug zhu
    #                     print(f"predicting {k},begin output next stage npy.")

    #                     resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
    #                                       self.dataset_json)

    #                     # results.append(segmentation_export_pool.starmap_async(
    #                     #     resample_and_save, (
    #                     #         (prediction, target_shape, output_file, self.plans_manager,
    #                     #          self.configuration_manager,
    #                     #          properties,
    #                     #          self.dataset_json),
    #                     #     )
    #                     # ))

    #                     # results.append(segmentation_export_pool.starmap(
    #                     #     resample_and_save, (
    #                     #         (prediction, target_shape, output_file, self.plans_manager,
    #                     #          self.configuration_manager,
    #                     #          properties,
    #                     #          self.dataset_json),
    #                     #     )
    #                     # ))
    #                     #debug zhu
    #                     print(f"predicting {k},finish output next stage npy.")

    #         # _ = [r.get() for r in results]
        

    #     if self.is_ddp:
    #         dist.barrier()

    #     if self.local_rank == 0:
    #         metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
    #                                             validation_output_folder,
    #                                             join(validation_output_folder, 'summary.json'),
    #                                             self.plans_manager.image_reader_writer_class(),
    #                                             self.dataset_json["file_ending"],
    #                                             self.label_manager.foreground_regions if self.label_manager.has_regions else
    #                                             self.label_manager.foreground_labels,
    #                                             self.label_manager.ignore_label, chill=True)
    #         self.print_to_log_file("Validation complete", also_print_to_console=True)
    #         self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]), also_print_to_console=True)

    #     self.set_deep_supervision_enabled(True)
    #     compute_gaussian.cache_clear()

    # =====================================================
    # # multi process
    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_gpu=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        
        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for k in dataset_val.keys():
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                 allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                     allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                output_filename_truncated = join(validation_output_folder, k)

                #debug zhu
                print(f"predicting {k},begin predict_sliding_window.")
                try:
                    prediction = predictor.predict_sliding_window_return_logits(data)
                except RuntimeError:
                    predictor.perform_everything_on_gpu = False
                    prediction = predictor.predict_sliding_window_return_logits(data)
                    predictor.perform_everything_on_gpu = True

                prediction = prediction.cpu()
                #debug zhu
                print(f"predicting {k},finish predict_sliding_window.")

                # this needs to go into background processes
                #debug zhu
                print(f"predicting {k},output seg nii.")
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )

                # results.append(
                #     segmentation_export_pool.starmap(
                #         export_prediction_from_logits, (
                #             (prediction, properties, self.configuration_manager, self.plans_manager,
                #              self.dataset_json, output_filename_truncated, save_probabilities),
                #         )
                #     )
                # )
                
                # for debug purposes
                # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                #              self.dataset_json, output_filename_truncated, save_probabilities)
                            

                #debug zhu
                print(f"predicting {k},finish output seg nii.")

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = nnUNetDataset(expected_preprocessed_folder, [k],
                                                num_images_properties_loading_threshold=0)
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file = join(output_folder, k + '.npz')

                         #debug zhu
                        print(f"predicting {k},begin output next stage npy.")

                        # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                        #                   self.dataset_json)

                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json),
                            )
                        ))

                        # results.append(segmentation_export_pool.starmap(
                        #     resample_and_save, (
                        #         (prediction, target_shape, output_file, self.plans_manager,
                        #          self.configuration_manager,
                        #          properties,
                        #          self.dataset_json),
                        #     )
                        # ))
                        #debug zhu
                        print(f"predicting {k},finish output next stage npy.")

            _ = [r.get() for r in results]

    

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]), also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

   