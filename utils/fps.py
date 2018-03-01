import time

class FPS:
    def __init__(self, cb_time, cb_func):
        self.start_time, self.end_time, self.full_time = 0, 0, 0
        self.show_time      = time.time()
        self.frame_cntr     = 0
        self.measure_time   = cb_time
        self.func           = cb_func

    def start(self):
        self.start_time     = time.time()

    def stop(self):
        self.full_time      += time.time() - self.start_time
        self.frame_cntr     += 1
        
        curr_time = time.time()

        if curr_time - self.show_time > self.measure_time:
            self.show_time              = curr_time

            fps = float(self.frame_cntr) / self.full_time
            self.func(fps)

            self.full_time, self.frame_cntr  = 0, 0
