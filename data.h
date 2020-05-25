/*
 * Copyright (C) 2015-2020 ETH Zurich and University of Bologna
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _INPUT_IMAGE_
#define _INPUT_IMAGE_

#define FRACTIONARY_BITS 7
#define DATA_TYPE 8
#define IMG_ROW 64
#define IMG_COL 64
#define IMG_DIM IMG_ROW*IMG_COL

#define FILT_WIN 3
#define FILT_DIM FILT_WIN*FILT_WIN

static uint8_t In_Img[4096] = { 
159, 153, 153, 151, 159, 173, 167, 118, 94, 104, 106, 108, 106, 114, 122, 124, 129, 131, 133, 135, 137, 135, 131, 131, 133, 133, 131, 131, 137, 129, 131, 131, 129, 129, 129, 126, 133, 126, 124, 104, 129, 149, 157, 151, 155, 153, 151, 159, 157, 155, 193, 217, 187, 104, 118, 122, 124, 120, 122, 122, 126, 124, 122, 124, 
157, 155, 159, 157, 165, 169, 157, 114, 92, 98, 106, 102, 102, 114, 120, 124, 129, 131, 131, 133, 133, 137, 131, 133, 126, 129, 129, 131, 129, 133, 133, 131, 129, 126, 126, 131, 131, 129, 126, 108, 102, 143, 161, 155, 159, 159, 151, 153, 151, 149, 149, 211, 221, 104, 116, 118, 122, 124, 126, 118, 129, 137, 54, 50, 
155, 159, 153, 157, 159, 163, 155, 120, 94, 104, 102, 100, 104, 112, 120, 129, 131, 129, 126, 129, 131, 129, 131, 129, 133, 129, 131, 129, 129, 131, 135, 129, 129, 129, 124, 124, 129, 129, 122, 112, 106, 131, 153, 163, 159, 157, 157, 155, 149, 151, 147, 179, 217, 203, 112, 118, 122, 122, 129, 129, 139, 52, 52, 46, 
155, 157, 157, 167, 165, 153, 165, 114, 86, 104, 106, 104, 106, 112, 120, 126, 131, 126, 124, 124, 133, 129, 124, 129, 131, 129, 122, 133, 129, 129, 129, 129, 135, 129, 129, 131, 126, 126, 129, 118, 108, 126, 149, 159, 163, 161, 159, 157, 155, 153, 149, 143, 207, 221, 145, 114, 120, 122, 124, 135, 48, 48, 48, 50, 
157, 163, 167, 165, 161, 159, 159, 120, 82, 104, 98, 102, 106, 112, 124, 129, 124, 133, 124, 124, 129, 129, 141, 133, 131, 129, 120, 124, 118, 120, 129, 129, 133, 129, 129, 126, 129, 129, 120, 118, 108, 126, 145, 157, 155, 157, 155, 153, 155, 157, 149, 145, 153, 219, 221, 96, 116, 124, 133, 42, 44, 50, 58, 44, 
159, 161, 161, 155, 163, 161, 159, 114, 84, 100, 102, 104, 104, 118, 114, 122, 124, 122, 129, 139, 124, 129, 129, 124, 122, 122, 126, 137, 139, 108, 122, 122, 122, 129, 129, 124, 126, 129, 124, 118, 104, 122, 143, 149, 153, 153, 153, 155, 157, 151, 143, 141, 145, 207, 223, 177, 108, 129, 46, 44, 54, 52, 48, 46, 
161, 167, 147, 137, 161, 163, 159, 118, 90, 98, 102, 104, 112, 116, 126, 124, 126, 124, 124, 131, 126, 129, 126, 141, 151, 145, 159, 175, 169, 171, 187, 193, 129, 124, 129, 126, 120, 124, 122, 114, 106, 124, 145, 145, 149, 153, 149, 149, 147, 145, 141, 141, 145, 143, 217, 225, 118, 46, 46, 46, 56, 50, 50, 44, 
167, 159, 120, 135, 169, 161, 159, 122, 74, 94, 92, 100, 106, 110, 122, 120, 126, 129, 124, 124, 179, 137, 122, 137, 131, 126, 145, 151, 161, 183, 177, 191, 191, 201, 116, 120, 122, 120, 116, 112, 104, 116, 145, 145, 149, 151, 153, 143, 143, 145, 141, 141, 141, 141, 195, 225, 209, 42, 44, 54, 44, 50, 42, 54, 
171, 143, 76, 129, 167, 165, 157, 116, 80, 90, 100, 98, 100, 108, 114, 120, 120, 122, 126, 124, 120, 118, 122, 131, 126, 133, 139, 169, 163, 179, 191, 197, 189, 201, 197, 110, 104, 114, 112, 108, 104, 118, 149, 151, 141, 137, 143, 149, 147, 143, 143, 143, 147, 145, 149, 209, 42, 40, 50, 56, 50, 44, 58, 28, 
159, 100, 72, 133, 165, 165, 159, 114, 76, 94, 94, 102, 106, 108, 120, 114, 122, 118, 122, 114, 108, 120, 129, 129, 129, 141, 133, 143, 181, 183, 181, 181, 195, 189, 175, 215, 227, 90, 106, 104, 104, 118, 153, 155, 147, 116, 129, 147, 147, 145, 143, 147, 147, 151, 159, 40, 40, 50, 50, 52, 46, 54, 38, 151, 
137, 82, 86, 137, 167, 167, 155, 112, 80, 94, 96, 110, 96, 112, 114, 116, 118, 126, 110, 104, 120, 124, 124, 129, 135, 149, 151, 141, 173, 177, 187, 169, 199, 209, 205, 213, 215, 201, 92, 100, 98, 112, 149, 151, 149, 104, 78, 139, 151, 145, 141, 147, 149, 153, 42, 40, 52, 54, 44, 50, 50, 80, 118, 151, 
88, 96, 88, 131, 161, 161, 157, 110, 80, 94, 94, 100, 102, 110, 116, 120, 118, 116, 110, 112, 122, 118, 126, 133, 141, 147, 147, 139, 171, 177, 201, 199, 211, 201, 201, 209, 211, 227, 60, 84, 90, 118, 151, 155, 155, 122, 44, 106, 147, 145, 139, 145, 151, 100, 38, 54, 50, 46, 66, 52, 66, 112, 149, 157, 
90, 90, 88, 131, 163, 165, 157, 114, 72, 92, 98, 100, 100, 110, 114, 116, 120, 106, 112, 108, 112, 124, 126, 131, 131, 141, 141, 141, 139, 199, 193, 201, 193, 185, 207, 207, 209, 211, 217, 155, 78, 116, 151, 153, 151, 114, 50, 60, 133, 143, 139, 147, 151, 38, 46, 56, 52, 56, 48, 58, 92, 143, 149, 165, 
90, 90, 88, 133, 159, 163, 157, 112, 80, 94, 96, 100, 98, 108, 116, 112, 143, 114, 108, 116, 124, 122, 137, 124, 141, 129, 131, 167, 185, 183, 199, 181, 201, 203, 211, 207, 207, 201, 213, 223, 82, 110, 151, 161, 151, 112, 46, 38, 88, 137, 137, 147, 46, 42, 54, 50, 54, 44, 50, 54, 153, 143, 163, 167, 
86, 92, 78, 129, 161, 163, 161, 114, 74, 84, 94, 96, 106, 106, 108, 92, 108, 102, 120, 120, 110, 137, 126, 124, 124, 110, 175, 193, 171, 175, 181, 195, 199, 197, 195, 205, 205, 205, 209, 213, 217, 96, 151, 153, 155, 110, 46, 36, 209, 181, 219, 211, 44, 50, 50, 52, 50, 56, 44, 147, 137, 167, 159, 157, 
94, 92, 84, 129, 165, 167, 157, 112, 80, 90, 108, 94, 98, 106, 120, 229, 98, 100, 104, 116, 131, 135, 131, 131, 102, 181, 161, 183, 185, 171, 195, 183, 191, 201, 197, 205, 201, 201, 207, 207, 217, 76, 145, 157, 151, 106, 34, 213, 197, 215, 215, 225, 42, 48, 50, 50, 54, 48, 129, 139, 161, 155, 159, 159, 
98, 98, 92, 131, 163, 165, 163, 114, 80, 94, 102, 100, 106, 108, 110, 201, 108, 102, 114, 110, 114, 131, 116, 96, 157, 145, 147, 171, 189, 181, 183, 169, 191, 191, 197, 195, 205, 197, 199, 201, 201, 209, 141, 153, 141, 96, 191, 207, 207, 201, 207, 225, 40, 50, 50, 48, 58, 46, 151, 153, 157, 159, 157, 153, 
98, 98, 96, 133, 163, 171, 167, 116, 72, 96, 98, 106, 94, 106, 94, 199, 116, 108, 110, 100, 120, 122, 126, 149, 143, 161, 155, 173, 173, 189, 179, 177, 171, 183, 193, 175, 187, 181, 197, 193, 199, 191, 203, 149, 215, 183, 209, 201, 203, 201, 211, 221, 46, 48, 42, 50, 56, 143, 145, 161, 159, 161, 153, 153, 
98, 96, 102, 129, 165, 177, 167, 114, 74, 98, 100, 96, 106, 106, 90, 171, 124, 110, 110, 112, 122, 114, 143, 157, 141, 157, 149, 179, 171, 179, 177, 179, 171, 179, 181, 177, 191, 191, 183, 185, 183, 163, 193, 187, 207, 197, 199, 199, 205, 187, 209, 217, 42, 52, 38, 54, 108, 143, 159, 159, 155, 161, 153, 157, 
92, 94, 98, 133, 169, 173, 167, 118, 76, 96, 106, 98, 100, 102, 90, 171, 129, 126, 82, 120, 114, 141, 161, 151, 147, 145, 139, 135, 175, 175, 159, 157, 163, 165, 179, 173, 171, 177, 179, 173, 173, 167, 193, 201, 193, 197, 197, 203, 187, 165, 179, 24, 46, 48, 52, 76, 149, 149, 159, 159, 157, 159, 159, 159, 
94, 102, 94, 135, 165, 167, 171, 120, 84, 94, 98, 96, 102, 100, 78, 179, 153, 141, 110, 129, 139, 141, 143, 153, 147, 157, 133, 151, 149, 159, 165, 155, 167, 173, 169, 173, 155, 155, 155, 179, 195, 199, 191, 195, 197, 203, 205, 209, 157, 165, 191, 46, 46, 42, 52, 122, 145, 157, 159, 157, 155, 155, 159, 157, 
100, 100, 106, 135, 169, 175, 169, 120, 74, 90, 96, 94, 98, 96, 62, 197, 181, 159, 118, 124, 145, 129, 135, 114, 149, 126, 147, 120, 96, 135, 163, 124, 169, 139, 153, 70, 157, 183, 169, 203, 189, 193, 195, 201, 205, 205, 209, 161, 80, 108, 233, 36, 44, 56, 60, 151, 153, 159, 155, 157, 153, 153, 159, 153, 
108, 98, 108, 143, 171, 173, 171, 122, 76, 94, 98, 104, 100, 94, 78, 205, 173, 129, 110, 124, 120, 133, 129, 135, 116, 143, 139, 94, 116, 94, 129, 76, 108, 94, 46, 38, 179, 197, 195, 195, 193, 197, 201, 199, 203, 209, 88, 82, 94, 201, 32, 54, 42, 48, 122, 143, 163, 161, 161, 157, 151, 153, 151, 159, 
104, 94, 110, 135, 169, 171, 165, 126, 72, 94, 96, 100, 96, 96, 80, 209, 189, 139, 145, 114, 139, 112, 135, 122, 135, 131, 135, 62, 66, 58, 116, 94, 74, 143, 34, 116, 197, 181, 191, 181, 189, 189, 197, 199, 129, 96, 100, 165, 193, 50, 50, 46, 60, 46, 149, 157, 167, 165, 157, 155, 153, 155, 155, 155, 
104, 108, 100, 135, 167, 171, 171, 126, 76, 86, 94, 104, 102, 102, 92, 221, 205, 181, 102, 135, 96, 133, 124, 124, 133, 90, 98, 44, 34, 133, 46, 92, 30, 48, 118, 199, 173, 183, 177, 181, 195, 199, 44, 124, 133, 165, 171, 199, 38, 54, 54, 42, 56, 88, 141, 165, 161, 165, 159, 157, 155, 155, 153, 153, 
108, 98, 94, 129, 167, 171, 169, 124, 74, 90, 98, 96, 94, 108, 110, 82, 199, 187, 124, 116, 131, 120, 135, 120, 88, 90, 38, 131, 46, 40, 58, 54, 50, 98, 197, 185, 175, 181, 181, 189, 179, 177, 66, 88, 124, 161, 203, 42, 48, 50, 48, 46, 50, 135, 145, 165, 163, 161, 163, 159, 155, 155, 153, 145, 
106, 102, 92, 129, 167, 171, 171, 129, 72, 90, 100, 94, 96, 102, 114, 98, 211, 189, 124, 114, 116, 131, 126, 62, 54, 80, 52, 98, 52, 48, 48, 78, 100, 163, 179, 177, 171, 173, 167, 197, 197, 199, 102, 46, 139, 147, 40, 42, 52, 56, 46, 56, 66, 133, 161, 167, 171, 163, 165, 159, 159, 153, 155, 151, 
110, 98, 96, 129, 169, 175, 173, 126, 76, 100, 100, 102, 98, 106, 112, 110, 221, 108, 114, 102, 124, 114, 44, 44, 42, 84, 68, 44, 88, 32, 70, 118, 122, 181, 173, 161, 177, 173, 193, 205, 203, 205, 139, 36, 64, 167, 46, 52, 50, 58, 46, 52, 129, 131, 149, 157, 165, 163, 159, 161, 157, 159, 153, 155, 
104, 102, 94, 129, 173, 175, 173, 131, 72, 92, 102, 104, 100, 143, 114, 116, 108, 112, 116, 131, 104, 52, 46, 38, 64, 54, 56, 122, 100, 30, 118, 185, 137, 189, 169, 153, 167, 185, 195, 203, 209, 211, 171, 54, 46, 169, 54, 54, 54, 48, 48, 48, 143, 159, 157, 157, 153, 151, 157, 161, 157, 155, 153, 149, 
104, 100, 94, 126, 163, 173, 173, 129, 76, 96, 102, 104, 100, 112, 112, 110, 133, 108, 141, 124, 68, 118, 48, 72, 38, 40, 52, 129, 36, 106, 161, 187, 165, 175, 151, 161, 171, 183, 195, 197, 207, 213, 197, 88, 40, 171, 56, 58, 62, 52, 58, 112, 135, 165, 161, 163, 161, 155, 147, 143, 145, 143, 155, 151, 
98, 100, 92, 126, 165, 175, 171, 131, 74, 90, 104, 100, 102, 116, 120, 118, 50, 42, 131, 70, 58, 129, 50, 50, 52, 70, 112, 42, 114, 171, 163, 195, 187, 92, 129, 167, 183, 175, 187, 193, 197, 205, 185, 60, 34, 175, 52, 60, 72, 50, 46, 137, 145, 161, 157, 157, 163, 159, 155, 151, 149, 143, 143, 139, 
102, 104, 94, 126, 165, 175, 179, 131, 80, 96, 102, 102, 98, 106, 112, 139, 46, 141, 86, 78, 52, 78, 50, 40, 34, 38, 42, 126, 185, 193, 155, 151, 124, 58, 133, 94, 110, 167, 185, 189, 179, 90, 122, 98, 42, 98, 48, 56, 52, 52, 62, 137, 159, 161, 159, 151, 161, 159, 147, 155, 157, 149, 143, 135, 
98, 94, 80, 116, 163, 179, 173, 139, 70, 90, 102, 92, 92, 185, 141, 155, 82, 36, 110, 58, 104, 50, 38, 38, 122, 38, 32, 143, 175, 193, 72, 50, 50, 46, 54, 54, 124, 147, 187, 207, 54, 52, 46, 56, 42, 54, 94, 58, 46, 50, 84, 133, 159, 151, 155, 157, 153, 153, 153, 149, 153, 145, 141, 137, 
100, 96, 80, 118, 167, 181, 175, 131, 80, 100, 98, 102, 88, 167, 197, 80, 38, 76, 66, 64, 157, 50, 38, 48, 60, 34, 116, 151, 197, 153, 110, 50, 54, 46, 205, 96, 116, 137, 197, 102, 96, 40, 86, 40, 46, 34, 72, 58, 50, 44, 139, 149, 157, 151, 153, 153, 151, 149, 147, 147, 143, 137, 126, 155, 
106, 94, 80, 126, 167, 181, 177, 137, 84, 104, 112, 42, 177, 131, 50, 124, 38, 36, 48, 86, 165, 46, 46, 46, 42, 26, 195, 189, 157, 133, 143, 137, 104, 131, 159, 159, 124, 129, 207, 171, 145, 131, 66, 56, 52, 46, 88, 60, 46, 54, 137, 157, 155, 151, 153, 153, 151, 149, 143, 139, 137, 161, 183, 187, 
88, 84, 68, 112, 167, 179, 181, 139, 80, 102, 108, 135, 120, 108, 36, 88, 74, 44, 86, 129, 112, 42, 44, 54, 38, 100, 181, 217, 112, 131, 157, 159, 155, 171, 161, 149, 143, 126, 189, 187, 155, 129, 118, 62, 48, 52, 147, 56, 48, 78, 131, 157, 151, 149, 147, 149, 145, 147, 143, 133, 189, 193, 191, 191, 
88, 76, 62, 118, 161, 181, 179, 137, 78, 104, 157, 139, 46, 60, 68, 86, 66, 44, 48, 131, 34, 124, 46, 44, 78, 193, 207, 68, 118, 141, 151, 171, 175, 177, 175, 147, 139, 131, 185, 197, 161, 135, 129, 58, 62, 40, 165, 42, 42, 129, 151, 155, 155, 153, 151, 145, 143, 141, 131, 185, 195, 195, 189, 197, 
78, 72, 64, 102, 167, 177, 173, 133, 86, 98, 110, 129, 66, 104, 58, 104, 60, 46, 50, 66, 34, 84, 34, 38, 114, 195, 40, 74, 118, 131, 149, 153, 175, 173, 165, 139, 137, 126, 167, 213, 155, 141, 129, 58, 46, 36, 137, 36, 56, 137, 199, 153, 151, 149, 145, 145, 145, 133, 175, 195, 193, 195, 205, 205, 
78, 74, 60, 100, 161, 169, 177, 139, 80, 104, 112, 62, 60, 52, 143, 116, 60, 54, 36, 124, 86, 185, 135, 44, 139, 84, 38, 74, 122, 129, 147, 155, 161, 169, 165, 139, 133, 122, 159, 225, 155, 141, 122, 42, 68, 30, 145, 30, 50, 139, 159, 151, 155, 147, 143, 147, 141, 133, 195, 191, 195, 207, 209, 201, 
82, 70, 56, 96, 165, 177, 179, 137, 70, 131, 116, 72, 129, 72, 40, 147, 147, 74, 34, 133, 171, 46, 34, 129, 145, 38, 36, 68, 118, 129, 141, 147, 157, 159, 153, 116, 141, 135, 147, 231, 145, 135, 106, 44, 78, 34, 159, 60, 114, 137, 153, 147, 149, 149, 149, 145, 135, 175, 193, 193, 207, 205, 207, 205, 
72, 78, 70, 120, 167, 173, 175, 139, 68, 96, 110, 96, 90, 157, 64, 88, 46, 40, 60, 40, 84, 52, 167, 189, 36, 42, 36, 82, 120, 131, 135, 145, 153, 159, 155, 139, 114, 112, 120, 175, 147, 131, 50, 60, 56, 56, 171, 58, 145, 153, 151, 151, 151, 149, 143, 143, 133, 193, 197, 205, 207, 209, 209, 207, 
80, 78, 64, 129, 163, 173, 179, 137, 78, 94, 149, 72, 52, 44, 84, 108, 98, 40, 124, 70, 126, 157, 98, 38, 48, 56, 48, 84, 122, 124, 137, 143, 147, 157, 157, 153, 143, 185, 199, 171, 145, 120, 46, 66, 58, 66, 151, 94, 147, 157, 155, 153, 153, 149, 147, 141, 131, 197, 197, 207, 207, 209, 209, 213, 
70, 72, 72, 124, 165, 171, 177, 147, 68, 122, 133, 112, 48, 66, 96, 80, 141, 129, 60, 100, 159, 80, 193, 54, 50, 76, 42, 102, 116, 126, 139, 139, 141, 149, 151, 145, 155, 209, 201, 165, 137, 44, 48, 64, 68, 62, 118, 72, 145, 153, 151, 149, 147, 145, 147, 133, 149, 197, 205, 207, 207, 209, 211, 211, 
58, 66, 70, 131, 169, 177, 179, 139, 66, 88, 131, 92, 62, 54, 126, 78, 122, 88, 155, 191, 187, 102, 48, 60, 46, 70, 42, 106, 110, 137, 126, 143, 145, 76, 129, 124, 108, 126, 74, 100, 139, 46, 50, 72, 72, 68, 44, 124, 149, 157, 153, 151, 149, 149, 147, 135, 157, 205, 209, 209, 209, 211, 213, 213, 
54, 58, 76, 131, 163, 175, 179, 137, 68, 133, 102, 143, 40, 54, 143, 90, 92, 112, 155, 169, 98, 32, 50, 46, 44, 54, 46, 80, 96, 124, 122, 133, 143, 133, 129, 124, 157, 173, 137, 149, 48, 50, 56, 54, 78, 58, 44, 157, 151, 153, 149, 155, 151, 145, 139, 135, 171, 213, 213, 211, 211, 215, 211, 209, 
52, 52, 88, 124, 159, 181, 179, 135, 64, 94, 48, 62, 54, 50, 145, 110, 50, 88, 102, 197, 114, 50, 54, 50, 56, 66, 52, 76, 70, 104, 118, 129, 137, 139, 133, 124, 114, 118, 149, 54, 40, 48, 52, 60, 102, 60, 52, 151, 155, 153, 149, 153, 149, 145, 143, 131, 175, 217, 213, 217, 213, 203, 207, 209, 
48, 52, 102, 129, 159, 177, 179, 139, 64, 104, 76, 52, 100, 62, 104, 135, 100, 126, 60, 217, 46, 50, 54, 54, 56, 60, 58, 66, 60, 80, 106, 133, 137, 141, 151, 151, 173, 155, 151, 54, 50, 70, 50, 58, 104, 76, 54, 151, 151, 153, 153, 151, 145, 145, 141, 126, 191, 217, 213, 213, 211, 211, 207, 203, 
52, 48, 90, 131, 163, 173, 179, 139, 64, 72, 44, 50, 82, 44, 135, 96, 131, 122, 80, 58, 50, 44, 64, 52, 44, 56, 56, 54, 52, 50, 62, 110, 126, 143, 151, 169, 167, 163, 131, 56, 54, 62, 48, 76, 106, 78, 54, 155, 147, 151, 151, 147, 145, 141, 139, 124, 205, 217, 211, 215, 207, 207, 191, 131, 
52, 44, 78, 114, 163, 171, 173, 131, 225, 155, 48, 64, 50, 50, 116, 80, 124, 143, 131, 169, 40, 52, 86, 60, 50, 50, 54, 52, 58, 68, 102, 131, 147, 139, 151, 159, 165, 173, 185, 50, 54, 60, 56, 84, 106, 92, 70, 157, 104, 124, 126, 135, 145, 141, 133, 114, 215, 209, 209, 205, 189, 92, 42, 58, 
88, 52, 64, 86, 163, 171, 175, 203, 64, 32, 34, 42, 56, 68, 66, 126, 151, 100, 100, 155, 30, 42, 72, 86, 48, 54, 52, 62, 54, 74, 114, 129, 131, 135, 143, 141, 145, 171, 197, 199, 203, 38, 38, 90, 92, 92, 54, 129, 131, 124, 114, 104, 106, 104, 112, 102, 211, 203, 209, 199, 98, 42, 76, 82, 
143, 110, 78, 74, 157, 169, 171, 139, 94, 40, 32, 124, 66, 90, 40, 56, 54, 64, 100, 133, 163, 183, 52, 90, 50, 48, 50, 60, 80, 56, 116, 126, 122, 137, 139, 147, 157, 177, 187, 195, 203, 207, 181, 48, 70, 82, 44, 141, 145, 139, 135, 124, 116, 102, 135, 68, 209, 213, 207, 84, 78, 74, 94, 82, 
131, 149, 110, 74, 161, 173, 177, 141, 137, 36, 58, 54, 56, 64, 48, 126, 72, 76, 104, 139, 84, 62, 44, 98, 62, 54, 58, 94, 104, 44, 131, 116, 133, 141, 139, 153, 161, 169, 181, 187, 195, 207, 211, 193, 46, 62, 100, 139, 145, 143, 141, 135, 126, 118, 129, 133, 211, 211, 175, 54, 90, 102, 100, 96, 
84, 159, 133, 58, 161, 179, 181, 145, 135, 48, 68, 42, 48, 42, 34, 38, 88, 96, 92, 129, 151, 92, 36, 80, 50, 48, 52, 60, 118, 40, 126, 133, 135, 139, 139, 149, 157, 163, 171, 185, 191, 201, 205, 223, 32, 46, 175, 151, 145, 145, 137, 137, 133, 126, 139, 185, 211, 209, 90, 96, 100, 112, 92, 104, 
48, 169, 145, 42, 157, 177, 181, 151, 54, 42, 52, 52, 54, 36, 40, 68, 48, 96, 76, 131, 126, 56, 40, 82, 56, 52, 44, 54, 94, 48, 120, 126, 137, 133, 139, 147, 157, 155, 167, 177, 189, 201, 209, 215, 219, 36, 145, 147, 143, 141, 135, 133, 126, 126, 197, 211, 213, 189, 86, 100, 112, 94, 106, 100, 
38, 163, 149, 40, 161, 175, 181, 149, 50, 42, 46, 36, 58, 46, 72, 38, 60, 68, 54, 135, 141, 171, 50, 60, 64, 54, 50, 100, 122, 42, 118, 131, 139, 137, 139, 141, 147, 159, 167, 173, 187, 201, 205, 215, 223, 30, 139, 141, 145, 139, 139, 131, 122, 165, 217, 211, 213, 133, 92, 106, 108, 92, 102, 88, 
34, 159, 155, 42, 161, 179, 179, 147, 70, 40, 56, 58, 46, 52, 48, 42, 36, 60, 96, 141, 102, 26, 34, 131, 60, 48, 64, 120, 131, 74, 131, 129, 143, 141, 135, 143, 151, 151, 159, 169, 183, 197, 203, 211, 215, 10, 141, 159, 141, 139, 137, 135, 122, 193, 205, 217, 197, 116, 100, 106, 92, 106, 92, 94, 
38, 151, 159, 58, 157, 177, 177, 141, 48, 42, 50, 56, 62, 48, 54, 44, 36, 102, 100, 135, 80, 100, 50, 58, 52, 54, 86, 126, 129, 98, 137, 141, 141, 139, 139, 137, 141, 151, 159, 163, 181, 195, 201, 209, 211, 213, 143, 147, 147, 143, 143, 131, 129, 139, 179, 217, 161, 112, 98, 94, 98, 102, 94, 96, 
30, 139, 165, 64, 159, 177, 177, 151, 82, 48, 60, 46, 64, 66, 48, 46, 44, 104, 84, 52, 56, 167, 68, 46, 40, 58, 106, 122, 66, 126, 131, 141, 145, 143, 137, 141, 143, 149, 149, 161, 173, 187, 199, 209, 211, 217, 88, 80, 110, 126, 137, 143, 155, 139, 181, 215, 129, 88, 100, 74, 100, 86, 94, 66, 
32, 145, 175, 52, 149, 177, 179, 155, 82, 50, 46, 72, 74, 70, 66, 52, 44, 86, 147, 78, 56, 46, 42, 40, 44, 84, 116, 116, 72, 135, 137, 139, 143, 141, 137, 133, 141, 145, 149, 155, 169, 185, 197, 207, 211, 219, 70, 98, 80, 62, 68, 88, 135, 167, 209, 193, 82, 74, 78, 108, 92, 98, 94, 54, 
88, 122, 185, 72, 151, 175, 179, 141, 94, 50, 54, 54, 52, 48, 68, 46, 40, 52, 62, 147, 108, 52, 38, 48, 70, 110, 124, 40, 124, 135, 141, 139, 139, 141, 143, 137, 141, 137, 145, 155, 161, 175, 191, 201, 209, 215, 187, 100, 100, 90, 60, 40, 187, 155, 211, 112, 54, 72, 102, 84, 82, 116, 78, 56, 
102, 131, 175, 96, 149, 177, 183, 153, 82, 46, 48, 52, 44, 80, 60, 44, 88, 108, 86, 124, 102, 40, 36, 46, 88, 120, 42, 112, 129, 137, 143, 139, 141, 137, 141, 141, 141, 137, 147, 157, 159, 165, 183, 199, 209, 213, 217, 96, 102, 92, 82, 78, 157, 205, 90, 52, 72, 90, 96, 72, 104, 98, 74, 62, 
66, 110, 175, 122, 147, 177, 187, 157, 40, 38, 56, 56, 54, 52, 48, 106, 38, 34, 76, 122, 52, 40, 58, 38, 76, 100, 120, 129, 129, 131, 141, 135, 137, 147, 147, 143, 141, 141, 139, 149, 161, 167, 179, 189, 205, 213, 215, 82, 102, 110, 100, 114, 114, 86, 78, 84, 88, 102, 64, 82, 114, 96, 64, 56, 
52, 70, 205, 157, 145, 177, 183, 106, 48, 44, 68, 58, 46, 137, 52, 126, 44, 80, 60, 72, 40, 56, 54, 94, 114, 116, 124, 126, 129, 133, 133, 141, 141, 139, 145, 145, 145, 143, 141, 147, 153, 163, 175, 183, 199, 207, 205, 60, 96, 104, 118, 114, 112, 100, 112, 82, 98, 68, 68, 118, 110, 64, 58, 56, 
46, 64, 199, 179, 143, 181, 181, 66, 48, 62, 56, 60, 54, 48, 52, 92, 90, 110, 58, 46, 54, 42, 84, 104, 118, 124, 122, 129, 129, 129, 135, 131, 139, 141, 145, 141, 143, 149, 147, 145, 157, 165, 169, 179, 193, 203, 211, 207, 88, 96, 112, 126, 120, 133, 100, 86, 86, 84, 106, 126, 78, 44, 58, 104
};

static uint8_t Filter_Kern[9] = { 4, 8, 4, 8, 16, 8, 4, 8, 4, };

static uint8_t Gold_Out_Img[4096] = {
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 78, 77, 78, 81, 82, 75, 60, 50, 50, 51, 51, 52, 56, 59, 62, 64, 65, 65, 66, 66, 66, 66, 65, 65, 64, 65, 65, 65, 65, 66, 65, 64, 63, 63, 64, 64, 64, 60, 56, 58, 68, 76, 78, 78, 78, 77, 76, 76, 76, 84, 97, 94, 73, 59, 59, 60, 61, 61, 62, 61, 53, 40, 0, 
0, 78, 78, 79, 80, 80, 74, 60, 50, 50, 51, 51, 52, 56, 59, 63, 64, 64, 63, 64, 65, 65, 64, 65, 65, 64, 64, 64, 64, 65, 65, 65, 64, 64, 63, 63, 64, 63, 61, 56, 57, 65, 75, 79, 79, 79, 78, 77, 76, 75, 78, 89, 98, 87, 67, 59, 60, 61, 63, 62, 53, 38, 27, 0, 
0, 79, 80, 81, 80, 79, 74, 60, 49, 49, 51, 51, 53, 56, 60, 63, 64, 63, 62, 63, 64, 64, 64, 65, 65, 64, 63, 63, 63, 63, 64, 65, 65, 64, 64, 63, 63, 63, 61, 58, 57, 63, 73, 78, 79, 79, 78, 77, 77, 76, 75, 81, 95, 97, 79, 62, 59, 61, 60, 52, 37, 27, 25, 0, 
0, 80, 81, 81, 80, 79, 74, 59, 48, 48, 51, 51, 53, 56, 60, 62, 63, 63, 63, 63, 64, 65, 65, 65, 64, 63, 62, 63, 62, 61, 62, 64, 64, 64, 64, 63, 63, 63, 61, 58, 57, 62, 71, 76, 78, 78, 77, 77, 77, 76, 74, 74, 86, 100, 92, 69, 59, 59, 51, 36, 26, 25, 25, 0, 
0, 80, 79, 78, 79, 80, 74, 59, 48, 48, 50, 51, 54, 57, 59, 61, 62, 62, 63, 64, 64, 64, 65, 65, 65, 65, 67, 70, 69, 66, 68, 68, 65, 63, 63, 63, 62, 62, 61, 57, 56, 62, 70, 74, 76, 76, 76, 76, 76, 75, 72, 71, 78, 94, 100, 83, 62, 49, 35, 25, 24, 25, 24, 0, 
0, 79, 74, 73, 78, 80, 74, 59, 48, 47, 49, 51, 54, 57, 60, 61, 62, 62, 63, 65, 67, 65, 64, 67, 68, 69, 73, 78, 79, 80, 83, 82, 75, 69, 65, 61, 61, 61, 60, 56, 55, 61, 69, 73, 74, 75, 75, 74, 74, 72, 71, 70, 73, 84, 99, 97, 69, 39, 25, 24, 25, 25, 24, 0, 
0, 74, 65, 68, 78, 81, 74, 59, 46, 45, 48, 50, 52, 56, 59, 60, 62, 62, 62, 65, 69, 66, 64, 66, 67, 68, 73, 79, 83, 88, 92, 93, 90, 84, 72, 61, 59, 59, 58, 55, 54, 60, 69, 73, 73, 74, 74, 73, 72, 71, 71, 71, 71, 77, 92, 96, 69, 34, 23, 24, 24, 24, 23, 0, 
0, 65, 54, 64, 78, 81, 74, 58, 45, 44, 48, 49, 51, 54, 57, 59, 60, 61, 61, 62, 63, 62, 62, 64, 65, 66, 71, 77, 84, 89, 92, 94, 96, 94, 84, 73, 65, 58, 55, 53, 54, 61, 70, 74, 71, 69, 70, 72, 72, 72, 71, 71, 72, 74, 80, 73, 47, 27, 24, 25, 24, 24, 26, 0, 
0, 55, 49, 63, 78, 81, 74, 57, 45, 45, 48, 50, 52, 54, 57, 58, 59, 60, 59, 57, 57, 60, 62, 64, 66, 68, 71, 76, 84, 90, 91, 92, 96, 97, 95, 94, 87, 68, 54, 51, 53, 60, 71, 75, 70, 62, 62, 69, 73, 72, 71, 72, 73, 71, 60, 41, 27, 23, 24, 25, 25, 28, 38, 0, 
0, 49, 48, 64, 78, 81, 73, 57, 45, 45, 48, 50, 52, 54, 57, 58, 59, 59, 57, 55, 58, 60, 62, 65, 68, 71, 72, 75, 83, 90, 92, 93, 97, 100, 100, 104, 103, 83, 56, 47, 50, 59, 71, 75, 70, 56, 51, 62, 71, 72, 71, 72, 72, 62, 40, 25, 23, 25, 25, 25, 29, 40, 55, 0, 
0, 46, 49, 64, 77, 80, 73, 57, 45, 45, 48, 50, 51, 54, 57, 58, 58, 57, 55, 55, 58, 60, 62, 65, 68, 71, 72, 73, 81, 90, 95, 97, 99, 100, 101, 104, 106, 93, 67, 50, 49, 58, 71, 76, 71, 54, 40, 50, 67, 71, 71, 72, 68, 48, 28, 24, 25, 26, 26, 28, 37, 54, 68, 0, 
0, 45, 49, 64, 77, 80, 73, 57, 44, 44, 48, 49, 51, 54, 56, 59, 59, 57, 55, 56, 58, 61, 63, 65, 67, 69, 70, 74, 81, 91, 97, 98, 98, 99, 101, 103, 104, 101, 90, 70, 54, 57, 71, 76, 71, 53, 34, 38, 57, 68, 70, 69, 56, 34, 25, 25, 26, 25, 26, 33, 49, 65, 75, 0, 
0, 44, 49, 63, 77, 80, 74, 57, 44, 44, 47, 49, 51, 53, 55, 57, 59, 57, 56, 57, 59, 62, 64, 64, 65, 66, 72, 81, 86, 91, 95, 96, 97, 99, 101, 103, 103, 103, 104, 91, 67, 60, 70, 77, 71, 52, 31, 32, 56, 73, 78, 69, 43, 26, 24, 25, 25, 25, 28, 42, 61, 73, 78, 0, 
0, 44, 48, 62, 77, 81, 74, 57, 44, 43, 47, 49, 50, 53, 57, 61, 58, 54, 56, 58, 61, 64, 64, 63, 62, 68, 79, 88, 89, 89, 92, 95, 97, 99, 100, 101, 102, 102, 104, 102, 84, 65, 68, 76, 71, 51, 34, 47, 76, 91, 96, 79, 41, 23, 25, 25, 25, 27, 38, 57, 71, 77, 79, 0, 
0, 45, 49, 63, 77, 81, 74, 58, 44, 45, 48, 49, 50, 53, 65, 74, 61, 51, 54, 57, 61, 64, 63, 61, 64, 73, 82, 88, 90, 90, 91, 93, 95, 98, 99, 100, 101, 101, 102, 104, 93, 73, 70, 75, 69, 54, 53, 76, 96, 102, 106, 87, 44, 23, 24, 25, 25, 33, 52, 69, 76, 78, 79, 0, 
0, 47, 51, 64, 78, 82, 75, 58, 45, 45, 49, 50, 51, 52, 66, 78, 65, 53, 54, 56, 59, 62, 61, 62, 69, 76, 79, 85, 90, 91, 90, 90, 92, 95, 97, 97, 98, 98, 99, 100, 97, 88, 79, 78, 74, 69, 79, 96, 102, 102, 106, 87, 44, 23, 24, 25, 29, 42, 63, 75, 78, 79, 78, 0, 
0, 48, 52, 65, 78, 83, 77, 58, 45, 45, 49, 50, 51, 51, 61, 74, 66, 55, 54, 55, 58, 61, 64, 68, 73, 76, 79, 84, 88, 90, 89, 88, 88, 91, 93, 93, 94, 95, 96, 96, 96, 94, 89, 87, 89, 90, 96, 101, 101, 101, 104, 86, 44, 23, 23, 26, 38, 57, 72, 78, 79, 79, 77, 0, 
0, 48, 53, 65, 79, 85, 78, 59, 45, 46, 49, 50, 50, 50, 58, 71, 67, 56, 53, 55, 59, 63, 70, 74, 74, 75, 77, 81, 86, 88, 87, 86, 85, 88, 89, 89, 91, 92, 92, 92, 90, 90, 92, 95, 98, 99, 99, 100, 98, 96, 94, 73, 38, 23, 24, 33, 51, 68, 76, 78, 79, 79, 78, 0, 
0, 48, 53, 66, 79, 85, 78, 60, 46, 46, 49, 49, 50, 49, 56, 70, 70, 60, 54, 57, 62, 67, 73, 75, 74, 73, 72, 75, 82, 84, 83, 81, 83, 85, 87, 87, 86, 87, 87, 88, 89, 90, 94, 97, 98, 99, 100, 98, 92, 89, 78, 48, 26, 23, 28, 43, 63, 74, 78, 79, 78, 78, 78, 0, 
0, 49, 53, 66, 79, 84, 78, 61, 46, 45, 48, 49, 49, 47, 54, 74, 78, 67, 59, 61, 66, 69, 71, 72, 73, 72, 70, 69, 72, 77, 78, 78, 80, 82, 80, 77, 79, 82, 85, 89, 93, 94, 96, 98, 99, 100, 100, 92, 79, 80, 73, 40, 22, 24, 34, 54, 71, 77, 78, 78, 77, 78, 78, 0, 
0, 50, 55, 68, 80, 85, 79, 61, 45, 44, 47, 48, 49, 45, 53, 78, 85, 72, 62, 62, 66, 67, 66, 66, 68, 69, 67, 62, 60, 66, 70, 68, 70, 69, 60, 58, 73, 86, 89, 94, 96, 97, 98, 99, 101, 98, 89, 73, 63, 70, 66, 37, 22, 27, 42, 63, 75, 78, 78, 78, 77, 77, 77, 0, 
0, 51, 56, 69, 81, 85, 79, 61, 45, 44, 48, 49, 49, 45, 56, 82, 87, 72, 62, 61, 63, 64, 64, 64, 65, 67, 63, 52, 48, 52, 57, 54, 54, 52, 40, 47, 76, 93, 94, 95, 96, 97, 98, 97, 93, 83, 68, 60, 61, 60, 45, 28, 23, 32, 53, 71, 79, 80, 79, 77, 76, 76, 77, 0, 
0, 51, 56, 68, 80, 85, 79, 61, 45, 44, 48, 49, 49, 46, 59, 86, 92, 76, 64, 61, 61, 62, 63, 64, 63, 63, 55, 41, 36, 42, 47, 43, 42, 42, 42, 59, 84, 93, 93, 93, 95, 92, 86, 83, 77, 68, 65, 68, 62, 44, 28, 24, 26, 38, 61, 76, 81, 81, 79, 77, 77, 76, 77, 0, 
0, 51, 54, 66, 79, 84, 79, 61, 45, 43, 47, 49, 50, 49, 59, 82, 94, 82, 66, 60, 60, 61, 63, 62, 58, 51, 44, 34, 31, 36, 38, 34, 32, 41, 61, 80, 89, 90, 91, 92, 94, 82, 62, 60, 67, 73, 77, 67, 45, 29, 24, 24, 30, 47, 67, 79, 81, 81, 80, 78, 77, 77, 76, 0, 
0, 50, 52, 65, 79, 84, 79, 61, 45, 43, 47, 48, 49, 51, 54, 68, 89, 86, 67, 59, 60, 62, 61, 54, 46, 40, 37, 37, 31, 28, 29, 31, 35, 54, 79, 89, 89, 88, 90, 92, 94, 79, 51, 46, 63, 75, 68, 45, 28, 25, 24, 24, 35, 56, 73, 80, 82, 81, 80, 79, 78, 77, 76, 0, 
0, 50, 52, 64, 79, 85, 80, 62, 45, 44, 48, 48, 49, 52, 53, 64, 85, 83, 65, 58, 59, 59, 51, 38, 34, 34, 35, 37, 31, 24, 29, 39, 52, 72, 86, 88, 87, 87, 90, 95, 97, 85, 56, 40, 55, 63, 45, 27, 25, 25, 25, 28, 42, 62, 75, 81, 82, 81, 81, 79, 78, 77, 76, 0, 
0, 50, 52, 65, 80, 86, 80, 62, 46, 45, 49, 49, 51, 54, 56, 65, 77, 70, 59, 57, 56, 48, 34, 26, 27, 32, 34, 37, 34, 28, 39, 55, 67, 81, 86, 84, 85, 89, 94, 99, 101, 93, 65, 37, 45, 56, 38, 24, 26, 26, 25, 33, 53, 68, 75, 79, 80, 80, 80, 79, 79, 78, 77, 0, 
0, 50, 52, 65, 80, 86, 81, 63, 46, 45, 50, 51, 53, 58, 58, 60, 63, 60, 59, 58, 50, 39, 28, 24, 26, 28, 34, 43, 39, 37, 56, 74, 79, 84, 83, 81, 84, 91, 96, 100, 103, 99, 75, 41, 40, 54, 41, 27, 27, 26, 27, 39, 61, 75, 78, 78, 78, 77, 77, 77, 77, 76, 76, 0, 
0, 49, 51, 64, 79, 85, 81, 63, 46, 45, 50, 51, 53, 57, 57, 55, 52, 52, 58, 53, 45, 41, 32, 26, 25, 27, 36, 43, 43, 53, 74, 86, 84, 78, 77, 80, 86, 91, 95, 99, 102, 101, 82, 46, 40, 54, 42, 29, 29, 27, 32, 49, 67, 77, 80, 79, 79, 77, 75, 74, 74, 74, 74, 0, 
0, 49, 51, 63, 78, 85, 81, 63, 46, 45, 50, 50, 51, 55, 58, 53, 43, 44, 51, 44, 39, 41, 33, 24, 24, 29, 37, 45, 58, 74, 82, 86, 77, 63, 65, 73, 80, 87, 93, 96, 95, 90, 74, 45, 38, 49, 39, 29, 29, 27, 35, 57, 73, 78, 79, 79, 79, 78, 76, 75, 74, 73, 72, 0, 
0, 48, 50, 62, 78, 86, 82, 64, 47, 45, 49, 49, 52, 59, 63, 56, 42, 42, 45, 38, 35, 35, 27, 23, 26, 26, 33, 55, 78, 84, 74, 66, 55, 45, 48, 56, 66, 80, 91, 90, 76, 61, 53, 39, 33, 39, 35, 28, 27, 27, 39, 61, 75, 78, 78, 78, 78, 78, 76, 76, 75, 73, 71, 0, 
0, 47, 47, 60, 78, 87, 82, 65, 47, 45, 49, 48, 55, 70, 73, 59, 41, 37, 40, 40, 41, 32, 22, 25, 30, 26, 36, 65, 86, 81, 57, 40, 32, 34, 43, 47, 57, 75, 87, 79, 53, 35, 33, 30, 26, 30, 33, 30, 25, 29, 46, 65, 76, 77, 77, 77, 77, 76, 75, 75, 74, 72, 70, 0, 
0, 46, 46, 60, 79, 87, 83, 65, 48, 47, 48, 47, 58, 72, 70, 51, 32, 29, 33, 44, 51, 35, 21, 25, 27, 31, 52, 76, 84, 75, 55, 39, 33, 44, 60, 59, 59, 74, 84, 73, 50, 37, 31, 26, 22, 26, 32, 30, 25, 33, 55, 71, 76, 76, 76, 76, 75, 74, 73, 72, 72, 72, 73, 0, 
0, 45, 45, 60, 79, 88, 83, 66, 50, 49, 49, 51, 61, 61, 50, 42, 30, 25, 32, 49, 54, 35, 22, 23, 23, 38, 72, 87, 79, 71, 66, 60, 56, 64, 75, 72, 65, 73, 86, 81, 67, 54, 41, 30, 24, 29, 37, 32, 26, 37, 60, 75, 76, 75, 75, 75, 74, 73, 72, 71, 75, 81, 85, 0, 
0, 41, 42, 58, 78, 88, 84, 67, 50, 50, 57, 57, 55, 45, 36, 37, 33, 26, 36, 51, 49, 34, 25, 24, 31, 56, 82, 82, 69, 68, 74, 76, 76, 79, 80, 75, 69, 73, 87, 89, 78, 65, 50, 34, 26, 35, 46, 36, 28, 44, 64, 75, 76, 75, 74, 74, 73, 72, 71, 76, 86, 92, 94, 0, 
0, 37, 39, 56, 77, 87, 84, 66, 49, 52, 62, 59, 44, 36, 36, 38, 33, 26, 34, 42, 39, 35, 27, 26, 46, 72, 74, 61, 59, 67, 75, 81, 84, 86, 82, 74, 69, 72, 86, 92, 81, 69, 55, 37, 26, 36, 49, 36, 32, 55, 73, 77, 76, 75, 74, 73, 71, 71, 75, 85, 94, 96, 97, 0, 
0, 36, 38, 54, 76, 86, 82, 66, 50, 50, 57, 51, 39, 37, 43, 43, 34, 25, 29, 35, 38, 43, 35, 31, 53, 65, 50, 43, 55, 66, 73, 79, 84, 85, 81, 73, 67, 69, 84, 93, 83, 70, 56, 36, 25, 34, 45, 33, 34, 62, 80, 79, 76, 74, 73, 72, 70, 72, 83, 93, 97, 99, 100, 0, 
0, 35, 36, 52, 74, 85, 82, 66, 50, 51, 51, 44, 38, 39, 47, 52, 42, 29, 30, 44, 53, 53, 44, 43, 54, 46, 31, 37, 55, 65, 71, 76, 80, 82, 78, 70, 66, 68, 83, 94, 83, 68, 53, 34, 26, 34, 45, 36, 39, 63, 77, 77, 75, 74, 73, 72, 71, 77, 89, 96, 98, 101, 102, 0, 
0, 35, 37, 53, 75, 86, 83, 65, 49, 52, 52, 45, 45, 43, 44, 52, 48, 32, 31, 48, 55, 48, 48, 56, 50, 31, 24, 37, 55, 64, 70, 74, 78, 79, 75, 67, 64, 65, 78, 90, 80, 64, 46, 32, 28, 38, 50, 46, 50, 67, 74, 75, 74, 74, 73, 71, 73, 83, 94, 98, 101, 103, 103, 0, 
0, 36, 40, 57, 77, 86, 83, 65, 48, 50, 53, 47, 46, 46, 42, 45, 40, 31, 32, 40, 47, 48, 55, 56, 39, 24, 25, 39, 55, 64, 68, 72, 76, 78, 75, 69, 66, 68, 77, 83, 76, 58, 39, 30, 29, 41, 55, 54, 61, 73, 75, 75, 75, 74, 72, 70, 74, 88, 98, 101, 103, 103, 104, 0, 
0, 37, 42, 60, 78, 85, 83, 66, 48, 51, 57, 46, 35, 37, 42, 46, 43, 39, 39, 44, 53, 60, 58, 43, 28, 25, 29, 42, 56, 63, 67, 71, 74, 76, 76, 73, 74, 83, 87, 83, 70, 49, 32, 29, 30, 41, 54, 56, 65, 76, 76, 75, 75, 74, 72, 70, 75, 90, 100, 102, 103, 104, 104, 0, 
0, 35, 42, 61, 78, 86, 83, 66, 48, 52, 59, 47, 32, 33, 43, 49, 54, 52, 51, 61, 66, 62, 52, 35, 27, 28, 31, 44, 57, 63, 67, 70, 70, 69, 71, 71, 75, 84, 83, 76, 61, 39, 28, 30, 33, 37, 46, 54, 67, 76, 76, 75, 74, 73, 72, 70, 78, 93, 101, 103, 104, 104, 105, 0, 
0, 32, 42, 62, 79, 86, 84, 65, 47, 50, 58, 49, 34, 36, 48, 50, 53, 57, 66, 76, 69, 50, 37, 30, 26, 28, 31, 43, 55, 62, 65, 68, 66, 61, 63, 65, 68, 71, 66, 61, 50, 32, 26, 31, 34, 32, 38, 56, 71, 76, 76, 75, 74, 73, 71, 71, 82, 97, 104, 104, 104, 105, 105, 0, 
0, 31, 42, 62, 79, 87, 84, 64, 47, 48, 51, 45, 32, 36, 52, 51, 46, 53, 70, 78, 60, 35, 25, 24, 25, 27, 29, 38, 49, 57, 62, 66, 67, 64, 63, 64, 67, 69, 63, 52, 37, 27, 26, 32, 35, 31, 37, 61, 75, 76, 75, 75, 74, 73, 70, 72, 85, 100, 106, 105, 105, 105, 105, 0, 
0, 30, 44, 62, 78, 87, 84, 64, 46, 43, 40, 36, 32, 38, 54, 53, 45, 48, 63, 73, 53, 29, 25, 25, 27, 28, 30, 34, 40, 49, 58, 64, 68, 68, 67, 67, 68, 71, 63, 43, 27, 25, 27, 33, 39, 34, 39, 63, 76, 76, 75, 75, 74, 72, 69, 72, 87, 102, 107, 106, 105, 104, 103, 0, 
0, 30, 46, 64, 78, 86, 84, 64, 44, 39, 33, 31, 34, 39, 52, 56, 53, 50, 54, 60, 43, 27, 26, 26, 27, 28, 29, 30, 33, 39, 50, 60, 66, 70, 72, 75, 76, 74, 61, 37, 26, 27, 28, 35, 42, 37, 41, 63, 75, 75, 75, 74, 73, 71, 69, 73, 90, 104, 107, 106, 105, 103, 100, 0, 
0, 29, 44, 63, 77, 85, 83, 69, 55, 44, 32, 29, 32, 37, 49, 55, 59, 58, 53, 48, 33, 26, 29, 28, 25, 26, 28, 28, 28, 32, 43, 57, 66, 71, 75, 80, 82, 80, 64, 38, 27, 28, 29, 37, 45, 40, 43, 63, 71, 71, 72, 72, 72, 70, 67, 73, 93, 105, 106, 104, 99, 89, 78, 0, 
0, 29, 39, 57, 76, 85, 84, 77, 67, 49, 30, 26, 28, 34, 45, 53, 60, 62, 60, 53, 33, 26, 33, 31, 26, 25, 26, 27, 28, 35, 47, 60, 67, 70, 74, 77, 81, 84, 75, 55, 40, 31, 29, 39, 47, 42, 45, 60, 65, 64, 64, 65, 66, 66, 62, 70, 93, 105, 104, 99, 81, 59, 45, 0, 
0, 36, 37, 51, 72, 84, 85, 77, 57, 34, 25, 28, 31, 33, 38, 48, 54, 53, 58, 59, 45, 36, 37, 35, 28, 25, 26, 28, 31, 38, 52, 62, 66, 68, 71, 73, 78, 86, 89, 84, 72, 51, 40, 40, 43, 40, 42, 57, 65, 63, 61, 59, 58, 58, 56, 65, 90, 104, 99, 83, 58, 39, 35, 0, 
0, 52, 43, 49, 70, 84, 84, 72, 48, 26, 25, 34, 35, 33, 33, 39, 41, 41, 52, 61, 59, 50, 39, 36, 30, 25, 28, 33, 35, 39, 52, 61, 64, 67, 70, 73, 78, 86, 92, 97, 95, 85, 70, 51, 39, 36, 42, 59, 69, 68, 65, 61, 57, 57, 56, 65, 90, 103, 89, 62, 43, 40, 42, 0, 
0, 64, 53, 50, 70, 85, 83, 73, 54, 32, 27, 30, 30, 28, 29, 35, 38, 40, 51, 61, 59, 45, 34, 35, 31, 26, 29, 38, 40, 39, 52, 62, 65, 68, 71, 74, 79, 84, 89, 94, 98, 101, 96, 72, 40, 34, 51, 66, 71, 71, 69, 66, 62, 61, 63, 75, 95, 99, 75, 49, 43, 47, 48, 0, 
0, 66, 59, 50, 69, 86, 85, 73, 52, 32, 26, 26, 24, 22, 23, 31, 38, 43, 50, 60, 57, 39, 29, 32, 30, 25, 27, 36, 40, 40, 52, 64, 66, 68, 70, 74, 78, 81, 86, 91, 96, 100, 103, 89, 53, 40, 61, 73, 72, 71, 69, 67, 64, 66, 75, 89, 100, 91, 63, 47, 49, 51, 50, 0, 
0, 65, 62, 49, 67, 86, 86, 69, 42, 26, 24, 24, 24, 22, 23, 26, 33, 39, 46, 59, 62, 45, 31, 31, 30, 25, 26, 36, 40, 39, 51, 64, 67, 68, 69, 73, 76, 79, 84, 89, 94, 99, 104, 101, 75, 49, 60, 73, 72, 70, 68, 66, 65, 73, 89, 101, 101, 82, 57, 49, 51, 50, 49, 0, 
0, 64, 62, 48, 67, 86, 85, 66, 37, 23, 23, 24, 24, 25, 25, 25, 27, 33, 43, 58, 62, 47, 34, 33, 32, 27, 31, 44, 47, 42, 53, 64, 68, 68, 69, 71, 75, 78, 82, 87, 93, 99, 103, 106, 85, 51, 56, 72, 71, 70, 68, 65, 67, 82, 99, 105, 96, 73, 54, 50, 51, 49, 48, 0, 
0, 63, 64, 50, 67, 86, 85, 66, 38, 24, 24, 26, 25, 25, 25, 22, 24, 34, 47, 58, 53, 38, 32, 36, 33, 28, 37, 53, 55, 50, 57, 66, 69, 69, 69, 70, 73, 76, 80, 85, 91, 97, 101, 105, 88, 61, 62, 73, 72, 70, 68, 66, 69, 83, 98, 103, 89, 66, 52, 50, 49, 49, 48, 0, 
0, 61, 65, 53, 68, 86, 84, 66, 39, 25, 25, 27, 28, 27, 24, 22, 26, 39, 49, 50, 46, 40, 34, 31, 28, 30, 43, 55, 56, 55, 63, 69, 70, 70, 69, 70, 72, 75, 78, 83, 89, 95, 100, 104, 99, 83, 69, 66, 67, 68, 69, 67, 69, 78, 92, 97, 80, 59, 49, 47, 48, 48, 46, 0, 
0, 60, 66, 55, 68, 85, 85, 68, 42, 27, 26, 28, 31, 30, 27, 23, 28, 42, 48, 40, 40, 44, 35, 24, 24, 34, 49, 53, 51, 57, 66, 69, 71, 70, 69, 69, 71, 73, 76, 80, 87, 93, 99, 103, 105, 92, 64, 51, 54, 57, 60, 64, 69, 77, 91, 91, 68, 50, 45, 45, 46, 47, 43, 0, 
0, 62, 68, 55, 67, 85, 85, 70, 46, 29, 26, 29, 32, 31, 29, 25, 27, 39, 48, 44, 38, 34, 27, 23, 28, 41, 51, 49, 49, 60, 68, 69, 70, 70, 69, 69, 70, 72, 74, 78, 84, 91, 97, 102, 105, 93, 64, 48, 45, 43, 42, 51, 69, 83, 92, 80, 53, 41, 43, 46, 47, 47, 42, 0, 
0, 64, 70, 59, 68, 85, 85, 70, 46, 29, 26, 27, 28, 30, 29, 26, 28, 36, 47, 53, 44, 28, 21, 24, 35, 47, 48, 45, 54, 65, 69, 69, 70, 70, 69, 69, 69, 70, 73, 77, 81, 87, 94, 100, 104, 100, 80, 57, 46, 41, 35, 44, 70, 85, 81, 60, 40, 39, 44, 44, 46, 48, 40, 0, 
0, 64, 71, 64, 71, 85, 86, 69, 43, 27, 25, 25, 26, 29, 30, 30, 32, 36, 44, 53, 44, 27, 21, 26, 39, 48, 46, 50, 61, 67, 69, 69, 69, 70, 70, 70, 70, 70, 72, 76, 80, 85, 91, 98, 103, 105, 90, 62, 49, 46, 41, 48, 68, 74, 59, 42, 38, 42, 43, 43, 47, 47, 38, 0, 
0, 58, 74, 72, 74, 85, 86, 65, 36, 24, 26, 27, 29, 32, 34, 36, 32, 31, 40, 44, 34, 25, 25, 31, 42, 51, 54, 59, 64, 66, 68, 69, 69, 71, 71, 71, 70, 70, 71, 75, 79, 83, 89, 95, 101, 105, 89, 59, 48, 51, 51, 54, 59, 56, 46, 40, 42, 42, 40, 45, 49, 44, 34, 0, 
0, 50, 78, 81, 77, 85, 81, 55, 30, 25, 28, 28, 31, 36, 38, 41, 37, 33, 35, 33, 27, 25, 32, 41, 50, 57, 60, 63, 64, 65, 67, 68, 69, 71, 72, 72, 71, 71, 71, 74, 77, 82, 87, 92, 98, 102, 90, 63, 49, 51, 55, 57, 56, 53, 48, 45, 43, 41, 43, 50, 48, 37, 31, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

#endif
