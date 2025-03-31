   if (idx >= n)
   17     return;
   18
   19   for (int stride = 1; stride < blockDim.x; stride *= 2) {
   20     int index = 2 * stride * tid;
   21     if (index < blockDim.x) {
   22       idata[index] += idata[index + stride];
   23     }
   24     __synthreads();
   25   }
   26   if (tid == 0)
   27     g_odata[blockIdx.x] = idata[0];
   28 }
   
