#pragma once

#include<stb_image.h>

class Image{
private:
    double* data_= nullptr;
    unsigned int height, width, nchannels;
public:
    Image(const char* filepath){
        int w, h, c;
        unsigned char* img_data= stbi_load(filepath, &w, &h, &c, 0);

        width= w, height= h, nchannels= c;
        for(unsigned int i=0, n= width*height*nchannels; i<n; i++){
            data_[i]= double(img_data[i])/255.0;
        }
    }

    
};