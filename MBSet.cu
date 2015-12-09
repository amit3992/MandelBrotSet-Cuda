/*
 * File:   MBSet.cu
 *
 * Created on November 25, 2015
 *
 * Purpose:  This program displays Mandelbrot set using the GPU via CUDA and
 * OpenGL immediate mode.
 *
 * Name: Amit Kulkarni
 * GTID: 903038158

 */
#include <iostream>
#include <stack>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "Complex.cu"
#include <GL/freeglut.h>

// Size of window in pixels, both width and height
#define WINDOW_DIM            512

using namespace std;

// Initial screen coordinates, both host and device.
Complex minC(-2.0, -1.2);
Complex maxC(1.0, 1.8);

//device copy of iterations array
int *deviceIter;

// To test Swap
int Ga = 10;
int Gb = 20;

const int maxIt = 2000; // Msximum Iterations

// Define the RGB Class
class RGB
{
public:
  RGB()
    : r(0), g(0), b(0) {}
  RGB(double r0, double g0, double b0)
    : r(r0), g(g0), b(b0) {}
public:
  double r;
  double g;
  double b;
};

RGB* colors = 0; // Array of color values

void InitializeColors()
{
  colors = new RGB[maxIt + 1];
  for (int i = 0; i < maxIt; ++i)
    {
      if (i < 5)
        { // Try this.. just white for small it counts
          colors[i] = RGB(1, 1, 1);
        }
      else
        {
          colors[i] = RGB(drand48(), drand48(), drand48());
        }
    }
  colors[maxIt] = RGB(); // black
}
// Function written in the Host but run in the device.
// 1] Calculate x & y index for every thread
// 2] iterate over all values (nIter < 2000)
// 3] copy all valid values to deviceIter
__global__ void calculateMBSet(int* deviceIter, double xD, double yD, double CminR, double CminI)
{

  // Step 1:
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int xIndex = index / WINDOW_DIM;
  int yIndex = index % WINDOW_DIM;

  Complex zo(0.0, 0.0);
  Complex c((CminR + xD * xIndex), (CminI + yD * yIndex));
  Complex z((CminR + xD * xIndex), (CminI + yD * yIndex));
  int it = 0;

  // Step 2:
  for(int i = 0; i < maxIt ; i++)
  {
    if(z.magnitude2() > 4.0) break;
      zo = z*z + c;
      z = zo;
      it++;
  }

  // Step 3:
  deviceIter[xIndex + yIndex * WINDOW_DIM] = it;

}

//Total size of iterations array
int winSize = sizeof(int) * WINDOW_DIM * WINDOW_DIM;

//Total result array of iterations
int *final = (int*) malloc(winSize);

void computeResult()
{
  // 1] Copy x & y values and calculate number of blocks required
  // 2] Create dynamic memory in Device memory (cudaMalloc)
  // 3] Call __global__ function which runs in the Device
  // 4] Copy the calculated values back into the Host memory (cudaMemcpy)
  // cout << "APPLE" << endl;
  // Step 1:
  double CminR = minC.r;
  double CminI = minC.i;
  double realD = maxC.r - minC.r;
  double imgD = maxC.i - minC.i;
  double xD = realD / WINDOW_DIM;
  double yD = imgD / WINDOW_DIM;
  int nBlocks = WINDOW_DIM * WINDOW_DIM/32;

  // Step 2:
  cudaMalloc((void **)&deviceIter, winSize);

  // Step 3:
  calculateMBSet <<< nBlocks,32 >>> (deviceIter, xD, yD, CminR, CminI);

  // Step 4:
  cudaMemcpy(final, deviceIter, winSize, cudaMemcpyDeviceToHost);
  //cout << "computeResult called " << endl;

}

// 1] Get value from final
// 2] Get random color value and plot
void plotMBSetPixel()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glBegin(GL_POINTS);
    for (int col = 0; col < WINDOW_DIM; col++)
    {
      for (int row = 0; row < WINDOW_DIM; row++)
      {
          // Step 1:
          int val = final[col + row * WINDOW_DIM];
          // Step 2:
          glColor3f(colors[val].r, colors[val].g, colors[val].b);
          glVertex2f(col, row);
      }
    }
  glEnd();

}

// Function to plot a sqaure using 4 given values
void plotSquare(int &a, int &b, int &c, int &d)
{

  glClear(GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
  gluOrtho2D(0, WINDOW_DIM, WINDOW_DIM, 0);
  glColor3f(1.0, 0.0, 0.0);

  glBegin(GL_LINE_LOOP);
    glVertex2f(a,b);
    glVertex2f(c,b);
    glVertex2f(c,d);
    glVertex2f(a,d);
  glEnd();

  glFlush();
  glutSwapBuffers();
}

// Function to swap to values
void SwapValue (int &a, int &b)
{
  int temp;
  temp = a;
  a = b;
  b = temp;

  return;
}

// Global stack declaration to maintain a history
stack<double> realCStack;
stack<double> imgCStack;

// 1] If 'q' is pressed, quit
// 2] If 'b' is pressed, check if both stacks arent empty.
// 3] Copy real and img values of maxC and minC from the stack
void keyboard (unsigned char key, int x, int y)
{
  // Use switch
  // Step 1:
  if(key == 'q')
  {
    exit(0);
  }

  // Step 2:
  if(key == 'b')
  {

    if( (!imgCStack.empty()) && (!realCStack.empty()))
    {
        // Step 3:
        maxC.i = imgCStack.top();
        imgCStack.pop();
        maxC.r = realCStack.top();
        realCStack.pop();
        minC.i = imgCStack.top();
        imgCStack.pop();
        minC.r = realCStack.top();
        realCStack.pop();
    }
    else
    {
      //cout << "INITIAL POSTION. CANNOT GO BACK!" << endl;
    }
    computeResult();
    glutPostRedisplay();
  }

}

// Global variables for mouse and zooming operation
bool flag = false; // check mouse button
int rMin, rMax, iMin, iMax;
int xStart, xEnd, yStart, yEnd; // For mouse operation

// 1] Get co-ordinate values when mouse is clicked. Min for button =  down. Max for button = up
// 2] Calculate factors if flag == true
// 3] Swap based on location
// 4] Reassign new values
// 5] Plot sqaure for zooming and recalculate. Reset flag for mouse click
void mouse(int button, int state, int x, int y)
{
  // Step 1:
  if(button == GLUT_LEFT_BUTTON && state == GLUT_UP)
  {
      rMax = x;
      iMax = y;
      flag = true;
  }

  if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
  {
      rMin = x;
      iMin = y;
      realCStack.push(minC.r);
      imgCStack.push(minC.i);
      realCStack.push(maxC.r);
      imgCStack.push(maxC.i);
  }

  if(flag)
  {
    // Step 2:
    xStart = rMin;
    yStart = iMin;
    xEnd = rMax;
    yEnd = iMax;

    double rDiff = maxC.r - minC.r;
    double rFactor = rDiff/(WINDOW_DIM-1);

    double iDiff = maxC.i - minC.i;
    double iFactor = iDiff/(WINDOW_DIM-1);

    int xDiff = xEnd - xStart;
    int yDiff = yEnd - yStart;

    // Step 3:
    if((yStart < yEnd) && (xStart < xEnd))
    {   // Right down. Don't swap
    }
    else if((yStart > yEnd) && (xStart > xEnd))
    {
        // up left
        // Swap both xStart and yStart
        SwapValue(xStart, xEnd);
        SwapValue(yStart, yEnd);
    }
    else if ((yStart < yEnd) && (xStart > xEnd))
    {   // down left
        // Swap xStart
        SwapValue(xStart, xEnd);
    }
    else if ((yStart > yEnd) && (xStart < xEnd))
    {
        // up right
        // swap yStart
        SwapValue(yStart, yEnd);
    }

    if(xDiff > yDiff)
    {
        xEnd = xStart + yDiff;
    }
    else if(xDiff < yDiff)
    {
        yEnd = yStart + xDiff;
    }

    // Step 4: Reassign values
    rMax = xEnd;
    rMin = xStart;
    iMax = yEnd;
    iMin = yStart;

    double rMinf = minC.r + (rFactor * rMin);
    double rMaxf = minC.r + (rFactor * rMax);
    double iMinf = minC.i + (iFactor * iMin);
    double iMaxf = minC.i + (iFactor * iMax);

    minC.r = rMinf;
    minC.i = iMinf;
    maxC.r = rMaxf;
    maxC.i = iMaxf;

    // Step 5:
    plotSquare(xStart, yStart, xEnd, yEnd);
    computeResult();
    flag = false;

    }
}

void display(void)
{
    glLoadIdentity();
    gluOrtho2D(0, WINDOW_DIM, WINDOW_DIM, 0);

    // Draw Mandelbrot
    plotMBSetPixel();

    // Swap the double buffers
    glutSwapBuffers();
}

void init(void)
{
  glEnable(GL_DEPTH_TEST);
  glShadeModel(GL_SMOOTH);
}

int main(int argc, char** argv)
{
  // Initialize OPENGL here
  // Set up necessary host and device buffers
  // set up the opengl callbacks for display, mouse and keyboard

  // Calculate the interation counts
  // Grad students, pick the colors for the 0 .. 1999 iteration count pixels
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(WINDOW_DIM, WINDOW_DIM);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("MandelBrot Set");
    glClearColor(1.0,1.0,1.0,1.0);
    computeResult();

    init();
    glMatrixMode(GL_MODELVIEW);
    InitializeColors();
    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutKeyboardFunc (keyboard);
    glutMouseFunc(mouse);


  glutMainLoop(); // THis will callback the display, keyboard and mouse
  return 0;

}
