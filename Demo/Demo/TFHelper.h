//
//  TFHelper.h
//  Demo
//
//  Created by zjcneil on 2019/1/30.
//  Copyright © 2019 zjcneil. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <opencv2/opencv.hpp>
#include <vector>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"


NS_ASSUME_NONNULL_BEGIN

@interface TFHelper : NSObject {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
}

+ (instancetype) sharedInstance;

- (BOOL) inferImage:(const cv::Mat &)inputImage
        resultImage:(cv::Mat&)result
            heatmap:(cv::Mat&)heatmap;
@end

NS_ASSUME_NONNULL_END
