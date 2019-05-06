import threading
import numpy as np
from time import time,sleep
from collections import deque
import signal
from core.config import cfg

DEBUG=True

sleep_duration=2e-5
force_stop=False

def sigint_handler(signum, frame):
    global force_stop
    force_stop = True
    cfg.CTRLC=True
    print('Catch CTRL-C. Force stop!')

#signal.signal(signal.SIGINT, sigint_handler)
# signal.signal(signal.SIGHUP, sigint_handler)
#signal.signal(signal.SIGTERM, sigint_handler)

class DataQueue(object):
    def __init__(self, maxlen=0):
        self.maxlen=maxlen
        self.q=deque(maxlen=maxlen)
        self.que_stop=False
        self.queue_lock=threading.Lock()

    def closed(self):
        return self.que_stop

    def close(self):
        self.q.clear()
        self.que_stop=True
    
    def open(self):
        self.que_stop=False

    def push(self, data):
        insert=False
        global force_stop
        while not self.que_stop and not force_stop:
            self.queue_lock.acquire()
            if len(self.q)<self.maxlen:
                self.q.append(data)
                insert=True
            self.queue_lock.release()
            if insert:
                break
            sleep(sleep_duration)
        return insert

    def pop(self):
        popout=False
        data=None
        global force_stop
        while not self.que_stop and not force_stop:
            self.queue_lock.acquire()
            if len(self.q)>0:
                #print('Comsumer consumes data')
                data=self.q.popleft()
                popout=True
            self.queue_lock.release()
            if popout:
                break
            sleep(sleep_duration)
        return data


class DataThread(threading.Thread):
    def __init__(self, data_reader=None, que=None, thread_id=0, workers_index=None, workspace_inds=None):
        super(DataThread, self).__init__()
        self.data_reader=data_reader
        self.que=que
        self.thread_id=thread_id
        self.workers_index=workers_index
        self.workspace_inds=workspace_inds

        self.thread_lock=threading.Lock()
        self.thread_stop=False
        self.insert=True

    def update(self, permute_inds):
        self.thread_lock.acquire()        
        self.permute_inds=permute_inds.copy()
        self.workers_index[self.thread_id]=0
        self.thread_lock.release()

    def notify_que(self):
        self.insert=True

    def _thread_function(self):
        global force_stop
        while not self.thread_stop and not force_stop: 
            if self.insert:
                self.thread_lock.acquire()
                index=self.workers_index[self.thread_id]
                roidb=self.data_reader.__getitem__(self.permute_inds[index])
                index=(index+1)%(self.workspace_inds[self.thread_id,1]-self.workspace_inds[self.thread_id,0]+1)
                self.workers_index[self.thread_id]=index  

                if len(roidb)>0:
                    self.insert=self.que.push(roidb)
                self.thread_lock.release()
            sleep(sleep_duration)
            
        print('Worker {} stopped'.format(self.thread_id))

    def stop(self):
        self.thread_stop=True

    def run(self):
        self._thread_function()

class DataLoader(object):
    def __init__(self, data_reader, shuffle=True, batch_size=1, num_workers=1):
        self.data_reader=data_reader
        self.num_images=self.data_reader.__len__()//batch_size*batch_size
        self.shuffle=shuffle
        self.batch_size=batch_size
        self.num_workers=num_workers
        
        self.threads=[]
        self.que=DataQueue(maxlen=self.batch_size)
        self.thread_stop=False

        self.debug=False
        self._shuffle()
        self.init_threads()
        
    def update_threads(self):
        for t in self.threads:
            t.update(self.permute_inds)

    def notify_threads(self):
        for t in self.threads:
            t.notify_que()

    def stop_join_threads(self):
        for t in self.threads:
            t.stop()
        for t in self.threads:
            t.join()

    def init_threads(self):
        self.assign_workspace()
        for i in range(self.num_workers):
            worker=DataThread(data_reader=self.data_reader, que=self.que, thread_id=i, workers_index=self.workers_index, workspace_inds=self.workspace_inds, )
            worker.update(self.permute_inds)
            self.threads.append(worker)

        for t in self.threads:
            t.start()            

    def stop(self):
        self.que.close()
        self.stop_join_threads()  

    def assign_workspace(self):
        size_per_thread=self.num_images//self.num_workers
        starts=size_per_thread*np.arange(self.num_workers)
        ends=size_per_thread+starts
        ends[-1]=max(ends[-1], self.num_images)
        self.workspace_inds=np.zeros((self.num_workers, 2), np.int32)
        self.workspace_inds[:,0]=np.array(starts)
        self.workspace_inds[:,1]=np.array(ends)
        self.workers_index=np.zeros(self.num_workers, np.int32)
    
    def __iter__(self):
        return self

    def _shuffle(self):        
        if self.shuffle:
            self.permute_inds=np.random.permutation(np.arange(self.num_images))
        else:
            self.permute_inds=np.arange(self.num_images)
        self.cur_index=0

    def __next__(self):
        roidbs=[]
        tic=time()

        if self.que.closed():
            self.que.open()
            self.notify_threads()

        data_num=self.batch_size
        while not self.thread_stop and not force_stop:
            roidbs.append(self.que.pop())            
            data_num-=1
            if data_num==0:
                toc=time()
                if self.debug:
                    print('DataLoader costs {}s'.format(toc-tic))
                break
            sleep(sleep_duration)
        self.cur_index+=self.batch_size
        if self.cur_index>=self.num_images or force_stop:
            self._shuffle()
            self.que.close()
            self.update_threads()            
            raise StopIteration
        else:
            #self._prepare_roidb(roidbs)
            return roidbs
