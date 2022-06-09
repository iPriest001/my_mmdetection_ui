import torch
import sys, cv2, time, os
import os.path as osp
from ui.object_detection_platform_rc import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QTabWidget
from PyQt5.QtWidgets import QApplication, QMainWindow,QMessageBox
import os
import prettytable as pt
from mmdet.apis import (inference_detector, init_detector)
from mmcv.utils import print_log
import mmcv
from mmcv import Config, DictAction
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.apis import multi_gpu_test
from PyQt5.Qt import QThread, QMutex, pyqtSignal
from mmcv.image import tensor2imgs
from mmdet.core import encode_mask_results
from ui.progress_ui_show import Progress_ui_show

qmut_1 = QMutex()  # thread lock
class Thread_1(QThread):  # thread 1 : detect single image
    detect_image_signal = pyqtSignal(list)
    def __init__(self, img, methods, dataset):  # img, methods, dataset, are string
        super().__init__()
        self.img = img
        self.methods = methods
        self.dataset = dataset

    def run(self):
        qmut_1.lock()
        self.detect_single_image()
        qmut_1.unlock()

    def detect_single_image(self):
        root_path = '/home/mst10512/mmdetection_219/'
        img = self.img  # current image
        show_score_thr = 0.5
        if self.methods == "Faster R-CNN + FPN":
            if self.dataset == "COCO":  # faster_rcnn_fpn_coco
                config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
                checkpoint = root_path + 'checkpoints/faster_rcnn/coco/faster_rcnn_r50_fpn_coco.pth'

            if self.dataset == "PASCAL VOC":  # faster_rcnn_fpn_voc
                config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc.py'
                checkpoint = root_path + 'checkpoints/faster_rcnn/voc/faster_rcnn_r50_fpn_voc.pth'


        if self.methods == "Faster R-CNN + CSA-FPN":
            if self.dataset == "COCO":  # faster_rcnn_fpn_csa_coco:
                config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_csa_1x_coco.py'
                checkpoint = root_path + 'checkpoints/faster_rcnn/coco/faster_rcnn_r50_fpn_csa_coco.pth'

            if self.dataset == "PASCAL VOC":  # faster_rcnn_fpn_csa_voc
                config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_csa_1x_voc.py'
                checkpoint = root_path + 'checkpoints/faster_rcnn/voc/faster_rcnn_r50_fpn_csa_voc.pth'


        if self.methods == "Faster R-CNN + AAR-FPN":
            if self.dataset == "COCO":  # faster_rcnn_fpn_aar_coco:
                config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_aar_1x_coco.py'
                checkpoint = root_path + 'checkpoints/faster_rcnn/coco/faster_rcnn_r50_fpn_aar2_coco.pth'

            if self.dataset == "PASCAL VOC":  # faster_rcnn_fpn_aar_voc
                config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_aar_1x_voc.py'
                checkpoint = root_path + 'checkpoints/faster_rcnn/voc/faster_rcnn_r50_fpn_aar2_voc.pth'


        if self.methods == "Faster R-CNN + CSA-FPN + AAR-FPN":
            if self.dataset == "COCO":  # faster_rcnn_csa_fpn_aar_coco:
                config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_csa_fpn_aar_1x_coco.py'
                checkpoint = root_path + 'checkpoints/faster_rcnn/coco/faster_rcnn_r50_csa_fpn_aar_coco.pth'
                show_score_thr = 0.6

            if self.dataset == "PASCAL VOC":  # faster_rcnn_csa_fpn_aar_voc
                config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_csa_fpn_aar_1x_voc.py'
                checkpoint = root_path + 'checkpoints/faster_rcnn/voc/faster_rcnn_r50_csa_fpn_aar_voc.pth'
                show_score_thr = 0.6


        if self.methods == "RetinaNet + FPN":
            if self.dataset == "COCO":  # retinanet_fpn_coco:
                config = root_path + 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
                checkpoint = root_path + 'checkpoints/retinanet/coco/retinanet_r50_fpn_coco.pth'

            if self.dataset == "PASCAL VOC":  # retinanet_fpn_voc
                config = root_path + 'configs/retinanet/retinanet_r50_fpn_1x_voc.py'
                checkpoint = root_path + 'checkpoints/retinanet/voc/retinanet_r50_fpn_voc.pth'


        if self.methods == "RetinaNet + CSA-FPN":
            if self.dataset == "COCO":  # retinanet_fpn_csa_coco:
                config = root_path + 'configs/retinanet/retinanet_r50_fpn_csa_1x_coco.py'
                checkpoint = root_path + 'checkpoints/retinanet/coco/retinanet_r50_fpn_csa_coco.pth'

            if self.dataset == "PASCAL VOC":  # retinanet_fpn_csa_voc
                config = root_path + 'configs/retinanet/retinanet_r50_fpn_csa_1x_voc.py'
                checkpoint = root_path + 'checkpoints/retinanet/voc/retinanet_r50_fpn_csa_voc.pth'


        if self.methods == "RetinaNet + AAR-FPN":
            if self.dataset == "COCO":  # retinanet_fpn_csa_coco:
                config = root_path + 'configs/retinanet/retinanet_r50_fpn_aar_1x_coco.py'
                checkpoint = root_path + 'checkpoints/retinanet/coco/retinanet_r50_fpn_aar2_coco.pth'

            if self.dataset == "PASCAL VOC":  # retinanet_fpn_csa_voc
                config = root_path + 'configs/retinanet/retinanet_r50_fpn_aar_1x_voc.py'
                checkpoint = root_path + 'checkpoints/retinanet/voc/retinanet_r50_fpn_aar2_voc.pth'


        self.object_detect_api(config, checkpoint, img, show_score_thr)  # detect image


    def object_detect_api(self, config, checkpoint, img, score_thr=0.5):  # config , checkpoint and img are both path (string)
        # build the model from a config file and a checkpoint file
        model = init_detector(config, checkpoint, device='cuda:0')
        begin_time = time.time()  # inference time start
        # test a single image
        result = inference_detector(model, img)
        end_time = time.time()
        consume_time = round((end_time - begin_time) * 1000, 5)  # en
        # save the results
        of = '/home/mst10512/mmdetection_219/detect_results/' + img.split('/')[-1]
        ui_result = model.show_result(img, result, score_thr=score_thr, show=False, out_file=of, ui_show=True)  # ui_show=True, return [img, ui_result]
        # show the detected image
        imagefile = of

        self.detect_image_signal.emit([consume_time, imagefile, ui_result])


class Thread_2(QThread):
    pushbuttun_signal = pyqtSignal()  # control push button
    method_info_signal = pyqtSignal(str)
    info_signal = pyqtSignal(str)
    inference_time_signal = pyqtSignal(list)
    mAP_signal = pyqtSignal(list)
    result_table = pyqtSignal(list)
    def __init__(self, baseline_method, new_method, dataset):
        super().__init__()
        self.baseline_method = baseline_method
        self.new_method = new_method
        self.dataset = dataset

    def run(self):
        self.detection_compare()
        self.pushbuttun_signal.emit()


    def detection_compare(self):
        root_path = '/home/mst10512/mmdetection_219/'
        if self.baseline_method == 'Faster R-CNN + FPN':
            if self.dataset == 'COCO':
                bl_config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
                bl_checkpoint = root_path + 'checkpoints/faster_rcnn/coco/faster_rcnn_r50_fpn_coco.pth'

            if self.dataset == 'PASCAL VOC':
                bl_config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc.py'
                bl_checkpoint = root_path + 'checkpoints/faster_rcnn/voc/faster_rcnn_r50_fpn_voc.pth'


        if self.baseline_method == 'RetinaNet + FPN':
            if self.dataset == 'COCO':
                bl_config = root_path + 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
                bl_checkpoint = root_path + 'checkpoints/retinanet/coco/retinanet_r50_fpn_coco.pth'

            if self.dataset == 'PASCAL VOC':
                bl_config = root_path + 'configs/retinanet/retinanet_r50_fpn_1x_voc.py'
                bl_checkpoint = root_path + 'checkpoints/retinanet/voc/retinanet_r50_fpn_voc.pth'


        if self.new_method == 'Faster R-CNN + CSA-FPN':
            if self.dataset == 'COCO':
                new_config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_csa_1x_coco.py'
                new_checkpoint = root_path + 'checkpoints/faster_rcnn/coco/faster_rcnn_r50_fpn_csa_coco.pth'

            if self.dataset == 'PASCAL VOC':
                new_config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_csa_1x_voc.py'
                new_checkpoint = root_path + 'checkpoints/faster_rcnn/voc/faster_rcnn_r50_fpn_csa_voc.pth'


        if self.new_method == 'RetinaNet + CSA-FPN':
            if self.dataset == 'COCO':
                new_config = root_path + 'configs/retinanet/retinanet_r50_fpn_csa_1x_coco.py'
                new_checkpoint = root_path + 'checkpoints/retinanet/coco/retinanet_r50_fpn_csa_coco.pth'

            if self.dataset == 'PASCAL VOC':
                new_config = root_path + 'configs/retinanet/retinanet_r50_fpn_csa_1x_voc.py'
                new_checkpoint = root_path + 'checkpoints/retinanet/voc/retinanet_r50_fpn_csa_voc.pth'


        if self.new_method == 'Faster R-CNN + AAR-FPN':
            if self.dataset == 'COCO':
                new_config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_aar_1x_coco.py'
                new_checkpoint = root_path + 'checkpoints/faster_rcnn/coco/faster_rcnn_r50_fpn_aar2_coco.pth'

            if self.dataset == 'PASCAL VOC':
                new_config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_aar_1x_voc.py'
                new_checkpoint = root_path + 'checkpoints/faster_rcnn/voc/faster_rcnn_r50_fpn_aar2_voc.pth'



        if self.new_method == 'RetinaNet + AAR-FPN':
            if self.dataset == 'COCO':
                new_config = root_path + 'configs/retinanet/retinanet_r50_fpn_aar_1x_coco.py'
                new_checkpoint = root_path + 'checkpoints/retinanet/coco/retinanet_r50_fpn_aar2_coco.pth'

            if self.dataset == 'PASCAL VOC':
                new_config = root_path + 'configs/retinanet/retinanet_r50_fpn_aar_1x_voc.py'
                new_checkpoint = root_path + 'checkpoints/retinanet/voc/retinanet_r50_fpn_aar2_voc.pth'


        if self.new_method == 'Faster R-CNN + CSA-FPN + AAR-FPN':
            if self.dataset == 'COCO':
                new_config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_csa_fpn_aar_1x_coco.py'
                new_checkpoint = root_path + 'checkpoints/faster_rcnn/coco/faster_rcnn_r50_csa_fpn_aar_coco.pth'

            if self.dataset == 'PASCAL VOC':
                new_config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_csa_fpn_aar_1x_voc.py'
                new_checkpoint = root_path + 'checkpoints/faster_rcnn/voc/faster_rcnn_r50_csa_fpn_aar_voc.pth'


        self.detect_dataset_api(bl_config, bl_checkpoint, use_method='baseline')
        self.detect_dataset_api(new_config, new_checkpoint, use_method='new_methods')

    def detect_dataset_api(self, config, checkpoint1, launcher='none', eval=True, use_method='baseline'):  # test
        info = str("[method]:  " + self.baseline_method)
        self.method_info_signal.emit(info)
        info = str("[test dataset]:  " + self.dataset)
        self.method_info_signal.emit(info)# send test method's details

        cfg = Config.fromfile(config)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        cfg.model.pretrained = None
        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None

        # in case the test dataset is concatenated
        samples_per_gpu = 1
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            samples_per_gpu = max(
                [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
            if samples_per_gpu > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        # init distributed env first, since logger depends on the dist info.
        if launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(launcher, **cfg.dist_params)

        rank, _ = get_dist_info()

        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # emit the number of test datasets
        test_img_nums = str(len(data_loader.dataset))
        test_img_nums_info = "Number of test dataset:    " + test_img_nums
        self.info_signal.emit(test_img_nums_info)

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, checkpoint1, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = self.single_gpu_test_ui_version(model=model, data_loader=data_loader, use_method=use_method)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model=model, data_loader=data_loader)

        rank, _ = get_dist_info()
        if rank == 0:
            if eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
                ]:
                    eval_kwargs.pop(key, None)
                #eval_kwargs.update(dict(metric=args.eval, **kwargs))
                if self.dataset=='COCO':
                    metric = dataset.evaluate(outputs, **eval_kwargs)
                    mAP = round(metric['bbox_mAP'], 5) * 100
                else:
                    metric, result_table = dataset.evaluate(outputs, ui_show=True, **eval_kwargs)
                    self.result_table.emit([result_table.table])
                    mAP = round(metric['mAP'], 5) * 100
                print(metric)
                metric_dict = dict(config=config, metric=metric)
                self.mAP_signal.emit([str(mAP), use_method])



    def single_gpu_test_ui_version(self,
                                   model,
                                   data_loader,
                                   use_method):
        model.eval()
        results = []
        dataset = data_loader.dataset
        nums_test_img = len(dataset)
        prog_bar = Progress_ui_show(nums_test_img)
        info1 = prog_bar.start()
        self.info_signal.emit(info1)

        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            batch_size = len(result)

            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            results.extend(result)

            for _ in range(batch_size):
                info2 = prog_bar.update()

                if (i % 200 == 0):
                    self.info_signal.emit(info2)
                if i == nums_test_img - 1:
                    self.info_signal.emit(info2[0])
                    time.sleep(0.5)
                    self.inference_time_signal.emit([info2[1], use_method])

        return results





class mywindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.listWidget_init()
        self.pushButton.clicked.connect(self.open_directory)
        self.listWidget_7.currentItemChanged.connect(self.image)  # open image from directory
        self.pushButton_2.clicked.connect(self.click_to_detect_single_image)
        self.pushButton_3.clicked.connect(self.click_to_compare_methods)


    def listWidget_init(self):
        self.listWidget_7.setSortingEnabled(True)


    def open_directory(self):
        self.listWidget_7.clear()  # refresh
        global filename1
        filename1=QFileDialog.getExistingDirectory(self,
                                    "选取文件夹",
                                    "./")
        if len(filename1):
            for i in os.listdir(filename1):
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap(filename1 +"/" +i).scaled(60, 30), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.listWidget_7.setIconSize(QtCore.QSize(60, 30))
                item = QtWidgets.QListWidgetItem(icon, filename1+"/"+i)
                self.listWidget_7.addItem(item)
        try:
            self.listWidget_7.setCurrentRow(0)
        except Exception as inst:
            print(inst)


    def image(self):  # load current image
        if self.listWidget_7.currentItem() is not None:
            imagefile=self.listWidget_7.currentItem().text()
            jpg = QtGui.QPixmap(imagefile).scaled(self.origin_img.width(), self.origin_img.height())
            self.origin_img.setPixmap(jpg)


    def click_to_detect_single_image(self):  # pushButton_2: detect_single_image ---- thread1
        def print_info(result_list):
            consume_time = result_list[0]
            imagefile = result_list[1]
            ui_result = result_list[-1]
            # consume time of inference
            self.lineEdit.setText(str(consume_time) + ' ms')
            # show image after detecting
            jpg = QtGui.QPixmap(imagefile).scaled(self.deteted_img.width(), self.deteted_img.height())
            self.deteted_img.setPixmap(jpg)
            # show detection details
            tb = pt.PrettyTable()
            tb.field_names = ['class', 'confidence', 'bbox left-top', 'bbox right-bottom']
            tb.align['class'] = 'l'
            tb.align['confidence'] = 'c'
            tb.align['bbox left-top'] = 'c'
            tb.align['bbox right-bottom'] = 'c'
            tb.border = False
            tb.padding_width = 5
            info = ui_result[1]
            for i in range(len(info)):
                row_data = [
                    info[i][0], f'{info[i][1]:.3f}', ('{:.3f}'.format(info[i][2]), '{:.3f}'.format(info[i][3]))
                    , ('{:.3f}'.format(info[i][4]), '{:.3f}'.format(info[i][5]))]
                tb.add_row(row_data)
            print_log(tb)
            text_data = str(tb)
            self.textBrowser.setText((text_data))

        try:
            if self.thread_1 != None:
                self.thread_1.terminate()
                time.sleep(1)
                self.listWidget.clear()  # refresh
                img = self.listWidget_7.currentItem().text()  # current image
                methods = self.comboBox.currentText()
                dataset = self.comboBox_2.currentText()
                self.thread_1 = Thread_1(img, methods, dataset)  # create thread
                self.thread_1.detect_image_signal.connect(print_info)
                self.thread_1.start()
        except Exception as inst:
            print(inst)
            self.textBrowser.clear()  # refresh
            img = self.listWidget_7.currentItem().text()  # current image
            methods = self.comboBox.currentText()
            dataset = self.comboBox_2.currentText()
            self.thread_1 = Thread_1(img, methods, dataset)  # create thread
            self.thread_1.detect_image_signal.connect(print_info)
            self.thread_1.start()


    def click_to_compare_methods(self):
        def set_btn():
            self.pushButton_3.setEnabled(True)

        def print_method_info(result):
            self.textBrowser_2.append("<font color='red'>" + result)

        def print_progress_info(result):
            self.textBrowser_2.append(result)

        def print_inference_time(result):
            if result[1] == 'baseline':
                self.lineEdit_4.setText(result[0] + ' s')
            else:
                self.lineEdit_6.setText(result[0] + ' s')

        def print_mAP(result):
            if result[1] == 'baseline':
                self.lineEdit_3.setText(result[0] + ' %')
            else:
                self.lineEdit_5.setText(result[0] + ' %')

        def print_table(result):
            table = result[0]
            self.textBrowser_2.append(table)


        try:
            if self.thread_2 != None:
                self.thread_2.terminate()
                time.sleep(1)
                self.textBrowser_2.clear()  # refresh
                baseline_methods = self.comboBox_3.currentText()
                new_methods = self.comboBox_4.currentText()
                dataset = self.comboBox_5.currentText()
                self.pushButton_3.setEnabled(False)
                self.thread_2 = Thread_2(baseline_methods, new_methods, dataset)
                self.thread_2.method_info_signal.connect(print_method_info)
                self.thread_2.info_signal.connect(print_progress_info)
                self.thread_2.inference_time_signal.connect(print_inference_time)
                self.thread_2.mAP_signal.connect(print_mAP)
                self.thread_2.result_table.connect(print_table)
                self.thread_2.pushbuttun_signal.connect(set_btn)
                self.thread_2.start()

        except Exception as inst:
            print(inst)
            self.textBrowser_2.clear()  # refresh
            baseline_methods = self.comboBox_3.currentText()
            new_methods = self.comboBox_4.currentText()
            dataset = self.comboBox_5.currentText()
            self.pushButton_3.setEnabled(False)
            self.thread_2 = Thread_2(baseline_methods, new_methods, dataset)
            self.thread_2.method_info_signal.connect(print_method_info)
            self.thread_2.info_signal.connect(print_progress_info)
            self.thread_2.inference_time_signal.connect(print_inference_time)
            self.thread_2.mAP_signal.connect(print_mAP)
            self.thread_2.result_table.connect(print_table)
            self.thread_2.pushbuttun_signal.connect(set_btn)
            self.thread_2.start()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = mywindow()
    myWin.show()
    sys.exit(app.exec_())






