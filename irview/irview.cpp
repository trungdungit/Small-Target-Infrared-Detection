#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

#ifdef linux
        #define CAMERA_INDEX 0
#else
        #define CAMERA_INDEX -1
#endif

// setup the camera index properly based on OS platform

// 0 in linux gives first camera for v4l
//-1 in windows gives first device or user dialog selection

#define PROG_ID_STRING "IrView - (c) Pham Trung Dung, 2020"
#define LICENSE_STRING "GPL - http://www.gnu.org/licenses/gpl.html"

static void print_help(char **name){
    printf("\n%s\n", PROG_ID_STRING);
	printf ("\t with OpenCV version %s (%d.%d.%d)\n",
					CV_VERSION,
					CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);
	printf("%s\n\n", LICENSE_STRING);


	printf("Usage :%s [image/video file]\n", name[0]);
	printf("Camera interface: run with no file agrument for direct camera use\n");
	printf("\nKeyboard commands\n");
	printf("\t a - automatic scaling (default: on)\n");
	printf("\t b - show both false colour and original (default: off)\n");
	printf("\t c - toggle false colour (default: on)\n");
	printf("\t e - exit (as per x or ESC)\n");
	printf("\t f - toggle full screen (default: off)\n");
	printf("\t x - exit\n\n");
}

//concat 2 opencv Mat objects side by side
Mat concatImage(Mat img1, Mat img2){
    Mat out = Mat(img1.rows, img1.cols+img2.cols, img1.type());
    Mat roi = out(Rect(0, 0, img1.cols, img1.rows));
    Mat roi2 = out(Rect(img1.cols, 0, img2.cols, img2.rows));

    img1.copyTo(roi);

    //depth of img1 is master, depth of img2 is slave
    //convert if needed

    if (img1.depth() != img2.depth()){
        // e.g. if img2 is 8-bit and img1 32-bit - scale to range 0->1 (32-bit)
        // otherwise img2 is 32-bit and img1 is 8-bit - scale to 0->255 (8-bit)
        img2.convertTo(roi2, img1.depth(), (img2.depth() < img1.depth()) ? 1.0 / 255.0 : 255);
	} else {
		img2.copyTo(roi2);
	}
	return out;
}

// set all mat values at given channel to given value
void setChannel(Mat &mat, int channel, unsigned char value){
    //make sure have enough channels
    if(mat.channels()<channel+1)
        return;
    //check mat is continuos
    if (mat.isContinuous())
        mat.reshape(1, mat.rows*mat.cols).col(channel).setTo(Scalar(value));
    else{
        for (int i=0;i<mat.rows;i++)
            mat.row(i).reshape(1,mat.cols).col(channel).setTo(Scalar(value));
    }
}

int main(int argc, char** argv){
    Mat img;
    VideoCapture capture;

    const char* windowNameHSV = PROG_ID_STRING;

    Mat HSV;
    bool keepProcessing = true;
    unsigned char key;
    int EVENT_LOOP_DELAY = 40;

    bool useFalseColour = true;		 
	bool useNormalisation = true;	 
	bool useConcatImage = false; 	 
	bool useFullScreen = false; 	 

    // if command line arguments are provided try to read image/video_name
	// otherwise default to capture from attached H/W camera

    if(
		( argc == 2 && (!(img = imread( argv[1], IMREAD_COLOR)).empty()))||
		( argc == 2 && (capture.open(argv[1]) == true )) ||
		( argc != 2 && (capture.open(CAMERA_INDEX) == true))
	)
    {
        print_help(argv);
        namedWindow(windowNameHSV, WINDOW_NORMAL);
        if (capture.isOpened()){
            capture >> img;
            if (img.empty()){
                if (argc ==2){
                    printf("End of video file reached/n");
                } else {
                    printf("ERROR: cannot get next frame from camera\n");
                }
                exit(0);
            }
        }
        resizeWindow(windowNameHSV, img.cols, img.rows);
        HSV = img.clone();
        Mat HSV_channels[3];

        while (keepProcessing){
            if (capture.isOpened()){
                capture >> img;
                if(img.empty()){
                    if (argc==2){
                        printf("End of video reached\n");
                    } else {
                        printf("ERROR: cannot get next frame from camera\n");
                    }
                    exit(0);
                }
            }
            //extract H,S,V channels
            split(img, HSV_channels);
            if (useNormalisation){
                normalize(HSV_channels[0],HSV_channels[0],0,255,NORM_MINMAX);
                normalize(HSV_channels[2],HSV_channels[2],0,255,NORM_MINMAX);
            }
            //set S channel to max
            setChannel(HSV_channels[1],0,255);
            // do scaling to avoid Hue space wrap around (i.e. dark == bright!)
			// N.B. changing the scaling factor and addition will vary the colour
			// effect - OpenCV 8-bit Hue in range 0->120 => 0.5 * Hue + 90 maps
			// all values to (wrap-around) 180->60 range in Hue.
            HSV_channels[0].convertTo(HSV_channels[0], -1 , 0.5, 90);
			merge(HSV_channels, 3, HSV);
			// put it all back together in RGB
			cvtColor(HSV, HSV, COLOR_HSV2BGR);
			// display image in window
            
            if (useConcatImage){
                imshow(windowNameHSV, concatImage(img, HSV));
            } else {
                if (useFalseColour){
                    imshow(windowNameHSV, HSV);
                } else {
                    imshow(windowNameHSV, img);
                }
            }
            key = waitKey(EVENT_LOOP_DELAY);
            switch (tolower(key)){
                case 'x':
                case 'e':
                case char(27): //ESC key
                    keepProcessing = false;
                    break;
                case 'a':
                    useNormalisation = (!useNormalisation);
                    break;
                case 'b':
					// toggle concatenated images
					useConcatImage = (!useConcatImage);
					break;
				case 'c':
					// toggle false colour
					useFalseColour = (!useFalseColour);
					break;
				case 'f':
                    // toggle false colour
                    useFullScreen = (!useFullScreen);
                    // set or unset the CV_WINDOW_FULLSCREEN flag via logical AND with toggle boolean
                    setWindowProperty(windowNameHSV, WND_PROP_FULLSCREEN, (WINDOW_FULLSCREEN & useFullScreen));
                    break;
            }
        }
        return 0;
    }
    print_help(argv);
    return -1;
}