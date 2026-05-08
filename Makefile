TEST_PRE_DIR=data/IXI_test_pre
TEST_IMG_DIR=$(TEST_PRE_DIR)/images
PREPROCESSING_LOGS=preprocessing/logs

build_dir: 
	mkdir -p $(TEST_IMG_DIR)/images/axial
	mkdir -p $(TEST_IMG_DIR)/images/coronal
	mkdir -p $(TEST_IMG_DIR)/images/sagittal

clean_axial:
	rm -rf $(TEST_IMG_DIR)/axial/* 

clean_coronal:
	rm -rf $(TEST_IMG_DIR)/coronal/* 

clean_sagittal:
	rm -rf $(TEST_IMG_DIR)/sagittal/* 

visual_clean:  clean_axial clean_coronal clean_sagittal

preprocess_clean: 
	rm -rf $(PREPROCESSING_LOGS)/*
