from shapescore import detection,maskgen,predict_survival,metrics
proj='example_detection'


if __name__ == '__main__':
    detection.predict_bb(proj)
    maskgen.predict_segmentation(proj)
    predict_survival.predict_survival(proj)
    metrics.metrics(proj)

