//
//  TFHelper.m
//  Demo
//
//  Created by zjcneil on 2019/1/30.
//  Copyright © 2019 zjcneil. All rights reserved.
//

#import "TFHelper.h"
#include <queue>
#include <iostream>

#import <AssertMacros.h>
#import <AssetsLibrary/AssetsLibrary.h>
// this package includes the definition of "float32_t"
#import <AVFoundation/AVFoundation.h>

#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/op_resolver.h"

#include <iostream>

#define LOG(x) std::cerr


static NSString* model_file_name = @"frozen_model";
static NSString* model_file_type = @"tflite";

static const int wanted_input_width = 600;
static const int wanted_input_height = 800;
static const int wanted_input_channels = 3;


@implementation TFHelper

static const int rec_min_width = 50;
static const int rec_min_height = 100;

static float32_t X[25][19] = {
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368},
    {-0.947368,-0.842105,-0.736842,-0.631579,-0.526316,-0.421053,-0.315789,-0.210526,-0.105263,0,0.105263,0.210526,0.315789,0.421053,0.526316,0.631579,0.736842,0.842105,0.947368}
};

static float32_t Y[25][19] = {
    {-0.96,-0.96,-0.96,-0.96,-0.96,-0.96,-0.96,-0.96,-0.96,-0.96,-0.96,-0.96,-0.96,-0.96,-0.96,-0.96,-0.96,-0.96,-0.96},
    {-0.88,-0.88,-0.88,-0.88,-0.88,-0.88,-0.88,-0.88,-0.88,-0.88,-0.88,-0.88,-0.88,-0.88,-0.88,-0.88,-0.88,-0.88,-0.88},
    {-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8},
    {-0.72,-0.72,-0.72,-0.72,-0.72,-0.72,-0.72,-0.72,-0.72,-0.72,-0.72,-0.72,-0.72,-0.72,-0.72,-0.72,-0.72,-0.72,-0.72},
    {-0.64,-0.64,-0.64,-0.64,-0.64,-0.64,-0.64,-0.64,-0.64,-0.64,-0.64,-0.64,-0.64,-0.64,-0.64,-0.64,-0.64,-0.64,-0.64},
    {-0.56,-0.56,-0.56,-0.56,-0.56,-0.56,-0.56,-0.56,-0.56,-0.56,-0.56,-0.56,-0.56,-0.56,-0.56,-0.56,-0.56,-0.56,-0.56},
    {-0.48,-0.48,-0.48,-0.48,-0.48,-0.48,-0.48,-0.48,-0.48,-0.48,-0.48,-0.48,-0.48,-0.48,-0.48,-0.48,-0.48,-0.48,-0.48},
    {-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4},
    {-0.32,-0.32,-0.32,-0.32,-0.32,-0.32,-0.32,-0.32,-0.32,-0.32,-0.32,-0.32,-0.32,-0.32,-0.32,-0.32,-0.32,-0.32,-0.32},
    {-0.24,-0.24,-0.24,-0.24,-0.24,-0.24,-0.24,-0.24,-0.24,-0.24,-0.24,-0.24,-0.24,-0.24,-0.24,-0.24,-0.24,-0.24,-0.24},
    {-0.16,-0.16,-0.16,-0.16,-0.16,-0.16,-0.16,-0.16,-0.16,-0.16,-0.16,-0.16,-0.16,-0.16,-0.16,-0.16,-0.16,-0.16,-0.16},
    {-0.08,-0.08,-0.08,-0.08,-0.08,-0.08,-0.08,-0.08,-0.08,-0.08,-0.08,-0.08,-0.08,-0.08,-0.08,-0.08,-0.08,-0.08,-0.08},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08},
    {0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16},
    {0.24,0.24,0.24,0.24,0.24,0.24,0.24,0.24,0.24,0.24,0.24,0.24,0.24,0.24,0.24,0.24,0.24,0.24,0.24},
    {0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.32},
    {0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4},
    {0.48,0.48,0.48,0.48,0.48,0.48,0.48,0.48,0.48,0.48,0.48,0.48,0.48,0.48,0.48,0.48,0.48,0.48,0.48},
    {0.56,0.56,0.56,0.56,0.56,0.56,0.56,0.56,0.56,0.56,0.56,0.56,0.56,0.56,0.56,0.56,0.56,0.56,0.56},
    {0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64},
    {0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72},
    {0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8},
    {0.88,0.88,0.88,0.88,0.88,0.88,0.88,0.88,0.88,0.88,0.88,0.88,0.88,0.88,0.88,0.88,0.88,0.88,0.88},
    {0.96,0.96,0.96,0.96,0.96,0.96,0.96,0.96,0.96,0.96,0.96,0.96,0.96,0.96,0.96,0.96,0.96,0.96,0.96},
};

+ (instancetype) sharedInstance {
    static TFHelper *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[self alloc] init];
    });
    return sharedInstance;
}

- (instancetype) init {
    if (self = [super init]) {
        [self loadModel];
    }return self;
}

static NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "." << [extension UTF8String]
        << "' in bundle.";
    }
    return file_path;
}

- (void) loadModel {
    NSString *graph_path = FilePathForResourceName(model_file_name, model_file_type);
    model = tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]);
    if (!model) {
        LOG(FATAL) << "Failed to mmap model " << graph_path <<std::endl;
    }
    LOG(INFO) << "Loaded Model:" << graph_path << std::endl;
    model->error_reporter();
    LOG(INFO) << "resolved reporter" << std::endl;
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter" << std::endl;
    }
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!" << std::endl;
    }
    LOG(INFO) << "Everything works well!" <<std::endl;
}

void dsnt(int height, int width, float32_t *heatmaps, int *X_coords, int *Y_coords) {
    // 1. norm the heatmaps.
    float32_t exp_sum[] = {0,0,0,0};
    int dim = 4;
    for (int h=0; h<height; h++) {
        auto a_row = heatmaps + h * dim * width;
        for (int w=0; w<width; w++) {
            auto a_pos = a_row + w * dim;
            for (int d=0; d<dim; d++) {
                auto the_element = a_pos[d];
                a_pos[d] = exp(the_element);
                exp_sum[d] += a_pos[d];
            }
        }
    }
    
    for (int d=0; d<dim; d++) {
        exp_sum[d] = MAX(exp_sum[d], 1e-12);
    }
    
    for (int h=0; h<height; h++) {
        auto a_row = heatmaps + h * dim * width;
        for (int w=0; w<width; w++) {
            auto a_pos = a_row + w * dim;
            for (int d=0; d<dim; d++) {
                a_pos[d] /= exp_sum[d];
            }
        }
    }
    
    // 2. coordinate transform
    float32_t x_coords[] = {0,0,0,0};
    float32_t y_coords[] = {0,0,0,0};
    
    for (int h=0; h<height; h++) {
        auto X_row = X[h];
        auto Y_row = Y[h];
        auto _row = heatmaps + h * width * dim;
        for (int w=0; w<width; w++) {
            auto x_element = X_row[w];
            auto y_element = Y_row[w];
            auto _ele = _row + w*dim;
            for (int d=0; d<dim; d++) {
                x_coords[d] += (_ele[d] * x_element);
                y_coords[d] += (_ele[d] * y_element);
            }
        }
    }
    
    for (int d=0; d<dim; d++) {
        auto x_coord = x_coords[d];
        auto y_coord = y_coords[d];
        
        int x_real_coord = (int)round( (x_coord+1)/2.0 * wanted_input_width);
        int y_real_coord = (int)round( (y_coord+1)/2.0 * wanted_input_height);
        
        X_coords[d] = x_real_coord;
        Y_coords[d] = y_real_coord;
    }
}

bool validate_coords(int *Xs, int *Ys) {
    /*
     */
    if (Xs[1] - Xs[0] <= rec_min_width) {
        return false;
    }
    if (Xs[3] - Xs[2] <= rec_min_width) {
        return false;
    }
    if (Ys[2] - Ys[0] <= rec_min_height) {
        return false;
    }
    if (Ys[3] - Ys[1] <= rec_min_height) {
        return false;
    }
    
    int upper_delta_y = Ys[0] - Ys[1];
    int lower_delta_y = Ys[2] - Ys[3];
    
    int left_delta_x = Xs[0] - Xs[2];
    int right_delta_x = Xs[1] - Xs[3];
    
    if (abs(upper_delta_y - lower_delta_y) > 100) {
        return false;
    }
    
    if (abs(left_delta_x - right_delta_x) > 100) {
        return false;
    }
    
    return true;
}

- (BOOL) inferImage:(const cv::Mat &)inputImage
        resultImage:(cv::Mat&)result
            heatmap:(cv::Mat&)heatmap {
    
    assert(inputImage.rows == wanted_input_height);
    assert(inputImage.cols == wanted_input_width);
    assert(inputImage.channels() == wanted_input_channels);
    assert(inputImage.type() == CV_32FC3);
    
    auto network_input = interpreter->inputs()[0];
    float32_t *network_input_ptr = interpreter->typed_tensor<float32_t>(network_input);
    const float *source_data = (float*) inputImage.data;
    
    std::memcpy(network_input_ptr, source_data, wanted_input_width*wanted_input_height*wanted_input_channels*sizeof(float32_t));
    
    /*
     *  float list -> cv::Mat
     *  https://stackoverflow.com/questions/22739320/how-can-i-initialize-a-cvmat-with-data-from-a-float-arrays
     *
     *  float dummy_query_data[10] = { 1, 2, 3, 4, 5, 6, 7, 8 };
     *  cv::Mat dummy_query = cv::Mat(2, 4, CV_32F, dummy_query_data);
     *
     *  cv::Mat -> float list
     *  float32_t *ptr = (float32_t *)inputImage.data;
     */
    
    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke!";
    }
    
    float32_t* network_output = interpreter->typed_output_tensor<float32_t>(0);
    
    auto output_h = 25;
    auto output_w = 19;
    auto output_dim = 4;
    
    int X_coords[4];
    int Y_coords[4];
    float32_t heatmaps[output_w*output_h*output_dim];
    
    std::memcpy(heatmaps, network_output, output_h*output_w*output_dim*sizeof(float32_t));
    
    dsnt(output_h, output_w, heatmaps, X_coords, Y_coords);
    
    cv::Mat input_dummy = inputImage.clone();
    
    
    if (validate_coords(X_coords, Y_coords)) {
        // there is a rectangle.
        auto lineColor = cv::Scalar(110, 220, 0);
        cv::line(input_dummy, cv::Point(X_coords[0], Y_coords[0]), cv::Point(X_coords[1], Y_coords[1]), lineColor, 5);
        cv::line(input_dummy, cv::Point(X_coords[1], Y_coords[1]), cv::Point(X_coords[3], Y_coords[3]), lineColor, 5);
        cv::line(input_dummy, cv::Point(X_coords[3], Y_coords[3]), cv::Point(X_coords[2], Y_coords[2]), lineColor, 5);
        cv::line(input_dummy, cv::Point(X_coords[2], Y_coords[2]), cv::Point(X_coords[0], Y_coords[0]), lineColor, 5);
    }{
        // there is no rectangle,
    }
    
    cv::Mat dummy = cv::Mat(output_h, output_w, CV_32FC4, heatmaps);
    std::vector<cv::Mat> heatmaps_activations(4);
    cv::split(dummy, heatmaps_activations);
    
    auto p1 = heatmaps_activations[0];
    auto p2 = heatmaps_activations[1];
    auto p3 = heatmaps_activations[2];
    auto p4 = heatmaps_activations[3];
    
    cv::Mat heatmaps_sum = p1+p2+p3+p4;
    
    cv::Mat _heatmap;
    cv::Mat grayscale;
    
    heatmaps_sum.convertTo(grayscale, CV_8UC3 , 255, 0);
    
    cv::applyColorMap(grayscale, _heatmap, cv::COLORMAP_JET);
    
    cv::Mat image_to_show;
    input_dummy.convertTo(image_to_show, CV_8UC3);
    
    result = image_to_show.clone();
    heatmap = _heatmap.clone();
    
    return NO;
    
}

@end
