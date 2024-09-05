#! /usr/bin/env python

'''
PullSeismicWaves.py
Purpose:  Multithread download of seismic waveforms with obspy
Author:   Christopher W Johnson, Los Alamos National Laboratory

Copyright 2024

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software. THE SOFTWARE IS 
PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE
'''

import sys
import argparse
import os
import copy
import datetime
import dateutil.relativedelta
import numpy as np 
import logging
import obspy
import obspy.clients.fdsn
import multiprocessing
import urllib.request
# Need for python >=3.8
multiprocessing.set_start_method("fork")

class Scheduler:
    '''
    Multithread scheduler
    Initiate workers, load params for each instance
    Build a queue of files for processing
    Go through the queue using all the workers
    '''
    def __init__(self, nproc, dataCenter):
        self._queue      = multiprocessing.Queue()
        self._nproc      = nproc
        self._dataCenter = dataCenter
        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for _ in range(self._nproc):
            self._workers.append(PullWave(self._queue, self._dataCenter))

    def start(self, fin):
        # put all into queue
        queue_count = 0
        for fin_ in fin:
            self._queue.put(copy.deepcopy(fin_))
            queue_count += 1
        #add a None to queue to indicate the end of task
        self._queue.put(None)
        print_str = f'\nAdd {queue_count} files to the queue'
        print(print_str)
        logging.info(print_str)
        print_str = f'Number of threads {self._nproc}'
        print(print_str)
        logging.info(print_str)

        # start the workers
        for worker in self._workers:
            worker.start()

        # and wait for all workers to finish
        for worker in self._workers:
            worker.join()


class PullWave(multiprocessing.Process):
    def __init__(self, queue, dataCenter):
        multiprocessing.Process.__init__(self, name='PullWave')
        self._queue = queue
        self.client = obspy.clients.fdsn.Client(dataCenter)

    def run(self):
        while True:
            params = self._queue.get()
            if params is None:
                self._queue.put(None)
                break
            else:
                self._pull_data(params)

    def _pull_data(self, param):
        dataCenter, NET, stat_, LOC, CHAN, t0, t1, outfile, \
        response, resp_out, pre_filt, taper = param

        if not response:
            try:
                st = self.client.get_waveforms(NET, stat_, LOC, CHAN, 
                    t0, t1, filename=outfile)
                print_str = f'Get : {outfile}'
                print(print_str)
                logging.info(print_str)
            except:
                print_str = f'Skip : {outfile}'
                print(print_str)
                logging.info(print_str)

        else:
            try:
                st = self.client.get_waveforms(NET, stat_, LOC, CHAN, 
                    t0, t1, attach_response=True)
                st.remove_response(output=resp_out, pre_filt=pre_filt, \
                    taper=taper, taper_fraction=0.001)
                st.write(outfile)
                print_str = f'Get : {outfile}'
                print(print_str)
                logging.info(print_str)
            except:
                print_str = f'Skip : {outfile}'
                print(print_str)
                logging.info(print_str)


class waveforms_list:
    def __init__(self, argv):
        super(waveforms_list, self).__init__()
        # Data Center
        self.datacenter = argv.datacenter

        # Directory
        self.cwd    = os.getcwd()
        self.outdir = os.path.join(self.cwd, argv.outdir)

        # Times
        self.tstart = obspy.UTCDateTime(argv.tstart)
        self.tend   = obspy.UTCDateTime(argv.tend)

        # Stations
        self.network  = argv.network.strip()
        self.stations = argv.stations.split()
        self.channel  = argv.channel.split()
        if len(self.channel) == 1:
            if len(argv.channel) == 2 and argv.remove_response:
                # SAC to multiple files
                self.channel = [argv.channel +'E',
                                argv.channel +'N',
                                argv.channel +'Z']
            elif len(argv.channel) == 2:
                # mseed to single file
                self.channel = [argv.channel +'*']

        self.loc      = argv.loc.strip()

        # Remove instrument response
        self.remove_response = argv.remove_response
        self.resp_out = argv.resp_out
        self.pre_filt = tuple(float(x) for x in argv.pre_filt.split())
        self.taper    = argv.taper
        self.dir_date = argv.dir_date
        self.dir_stat = argv.dir_stat
        # Parallel
        self.parallel = argv.parallel

        # Task
        self.download = argv.download
        self.response = argv.response


        # Formatting
        self.t_fmt = '%Y-%m-%dT%H_%M_%S'
        if self.remove_response:
            # removing response creates float64, must save as sac file
            self.fout  = '{net}.{stat}.{loc}.{chan}_{YY}{DOY:03d}_vel.sac'
        else:
            self.fout  = '{net}.{stat}.{loc}.{chan}_{YY}{DOY:03d}.mseed'

        # Set up DL
        self._build_dl_list()

    def _build_dl_list(self):
        # station_list
        if len(self.stations) == 1 and os.path.isfile(self.stations[0]):
            with open(self.stations[0], 'r') as f:
                 network_list = []
                 station_list = []
                 for line in f:
                    tmp = line.strip().split()
                    network_list.append(tmp[0])
                    station_list.append(tmp[1])
        else:
            station_list = self.stations
            network_list = [None]*len(station_list)

        secs_per_day = 86400
        t0    = copy.deepcopy(self.tstart)
        self.dl_params = []
        while t0 < self.tend:
            t1 = copy.deepcopy(t0 + secs_per_day)
            # if self.dir_date:
            #     out_dir = os.path.join(self.outdir, str(t0.datetime.year), 
            #                            '{0:03d}'.format(self._DOY(t0.datetime)))
            # else:
            #     out_dir = self.outdir

            # if self.download:
            #     does_dir_exist(out_dir)

            for stat_, net_ in zip(station_list, network_list):
                out_dir = self.outdir
                if self.dir_stat:
                    out_dir = os.path.join(out_dir, stat_)

                if self.dir_date:
                    out_dir = os.path.join(out_dir, str(t0.datetime.year), 
                                       '{0:03d}'.format(self._DOY(t0.datetime)))
                if self.download:
                    does_dir_exist(out_dir)

                if net_ is None:
                    net_ = self.network.upper()
                
                for chan_ in self.channel:
                    fout_ = self.fout.format(net=net_,
                                             stat=stat_.upper(), 
                                             loc=self.loc,
                                             chan=chan_.upper(),
                                             YY=t0.datetime.year,
                                             DOY=self._DOY(t0.datetime))

                    outfile = os.path.join(out_dir, fout_)
                    if os.path.isfile(outfile) and not self.response:
                        print(f'Exist: {fout_}')
                        if os.path.getsize(outfile) < 4096:
                            print('Redownload')
                        else:
                            continue
                    self.dl_params.append([self.datacenter, 
                        net_, stat_.upper(),
                        self.loc, chan_.upper(), 
                        t0, t1, outfile, 
                        self.remove_response, self.resp_out, self.pre_filt, self.taper])
            # Next day
            t0 += secs_per_day


    def _DOY(self, dt):
        return dt.timetuple().tm_yday


def does_dir_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def print2screen(args):
    prt_str = '{0} {1} {2} {3} {4} {5} {6} {7}'
    print(f'Total attempted downloads :: {len(args)}')
    for args_ in args:
        print(prt_str.format(*args_[:8]))


def get_response(args, outpath):
    outdir = os.path.join(outpath, 'resp')
    does_dir_exist(outdir)
    url = 'http://service.iris.edu/fdsnws/station/1/query?net={net}&sta={sta}&loc=**&cha=*H*&starttime=1995-01-01T00:00:00&level=response&format=xml&nodata=404'
    # url = 'http://service.iris.edu/fdsnws/station/1/query?net={net}&sta={sta}&loc=**&cha={chan}&starttime=1995-01-01T00:00:00&level=response&format=xml&nodata=404'
    # url = 'http://service.iris.edu/irisws/resp/1/query?net={net}&sta={sta}&loc=??&cha={chan}&starttime=1990-01-01T00:00:00'
    resp_out = 'RESP_{net}_{sta}_{chan}.xml'

    tmp = []
    args_reduce = []

    for args_ in args:
        net  = args_[1]
        sta  = args_[2]
        chan = args_[4]
        # one_str = net+sta+chan 
        one_str = net+sta 
        if one_str in tmp:
            continue
        tmp.append(one_str)
        args_reduce.append([net, sta, chan])

    for args_ in args_reduce:
        net  = args_[0]
        sta  = args_[1]
        chan = args_[2]
        resp_out_ = os.path.join(outdir, resp_out.format(net=net,sta=sta,chan=chan))
        if os.path.isfile(resp_out_):
            print('File exists')
            continue
        else:
            try:
                url_data = urllib.request.urlopen(url.format(net=net,sta=sta))
                with open(resp_out_, 'w') as f:
                    print(url_data.read().decode('utf-8'), file=f)
            except:
                print(f'Not downloaded :: ' + url.format(net=net,sta=sta))


def cmdLineParser(iargs=None):
    parser = createParser()
    return parser.parse_args(iargs)


def createParser():
    # Parse command line
    description = "Download seismic waveforms with obspy Client interface"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-o", "--outdir",
                        type=str, 
                        default="mseed",
                        help="Output directory. Default 'mseed'" )

    parser.add_argument("--datacenter", 
                        type=str, 
                        default="IRIS",
                        help="Data Center. See obspy Basic FDSN Client Usage for listing" )

    parser.add_argument("-t0", "--tstart",
                        type=str,
                        default="2014-01-01T00:00:00",
                        help="YYYY-MM-DDThh:mm:ss OR YYYY-MM-DD")

    parser.add_argument("-t1", "--tend",
                        type=str,
                        default="2014-01-02T00:00:00",
                        help="YYYY-MM-DDThh:mm:ss OR YYYY-MM-DD")

    parser.add_argument("--network",
                        type=str,
                        default="BK",
                        help="Station network. Default BK")

    parser.add_argument("--stations",
                        type=str,
                        default="BKS PKD",
                        help="Stations to download. Default 'BKS PKD'.\
                        Also accepts file with list of <net sta>")

    parser.add_argument("--channel",
                        type=str,
                        default="HH",
                        help="Channel default is HH; will download ENZ to single file. \
                        Or list single components; HHN, HHE, HHZ, will download ENZ to 3 files")

    parser.add_argument("--loc",
                        type=str,
                        default="*",
                        help="Station location. Default *")

    parser.add_argument("--parallel",
                        type=int,
                        default=4,
                        help="Parallel download")

    parser.add_argument("--remove_response",
                        default=False,
                        action='store_true',
                        help="Remove instrument response. Outfile in SAC format")

    parser.add_argument("--resp_out",
                        type=str,
                        default="VEL",
                        help="Remove response output: [VEL, DISP]")

    parser.add_argument("--pre_filt",
                        type=str,
                        default="0.005 0.006 40.0 45.0",
                        help="Remove response filter corners. Default '0.005 0.006 40.0 45.0'")

    parser.add_argument("--taper",
                        default=False,
                        action='store_true',
                        help="Apply default taper when remove response")

    parser.add_argument("--dir_date",
                        default=False,
                        action='store_true',
                        help="Use YYYY/DOY directory structure.")

    parser.add_argument("--dir_stat",
                        default=False,
                        action='store_true',
                        help="Use <stat>/*.mseed directory structure.")

    parser.add_argument("--print",
                        default=False,
                        action='store_true',
                        help="Print list of stations to download")

    parser.add_argument("--download",
                        default=False,
                        action='store_true',
                        help="Download stations")

    parser.add_argument("--response",
                        default=False,
                        action='store_true',
                        help="Download response file")

    return parser


def main(args):
    tic  = datetime.datetime.now()
    argv = cmdLineParser(args[1:])

    if not any([argv.print, argv.response, argv.download]):
        print("Enter functin to perform")
        print("--print, --response, or --download")
        sys.exit()

    # Build waveform list
    waveList = waveforms_list(argv)
    if argv.print:
        print2screen(waveList.dl_params)

    elif argv.response:
        print('Get response files')
        get_response(waveList.dl_params, argv.outdir)

    elif argv.download:
        # Logger
        time_now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        flog = 'PullSeismicWaves.log_' + time_now + '.txt'
        logging.basicConfig(filename=flog, level=logging.INFO, format='%(message)s')
        logging.info(' '.join(args))
        logging.info('outdir          - {0}'.format(argv.outdir))
        logging.info('datacenter      - {0}'.format(argv.datacenter))
        logging.info('tstart          - {0}'.format(argv.tstart))
        logging.info('tend            - {0}'.format(argv.tend))
        logging.info('network         - {0}'.format(argv.network))
        logging.info('stations        - {0}'.format(argv.stations))
        logging.info('channel         - {0}'.format(argv.channel))
        logging.info('loc             - {0}'.format(argv.loc))
        logging.info('remove_response - {0}'.format(argv.remove_response))
        logging.info('resp_out        - {0}'.format(argv.resp_out))
        logging.info('pre_filt        - {0}'.format(argv.pre_filt))
        logging.info('taper           - {0}'.format(argv.taper))
        logging.info('dir_date        - {0}'.format(argv.dir_date))
        logging.info('dir_stat        - {0}'.format(argv.dir_stat))

        # Build scheduler for N processes
        dl_sched = Scheduler(waveList.parallel, waveList.datacenter)
        # Start downloads
        dl_sched.start(iter(waveList.dl_params))

        # Wrap it up
        toc  = datetime.datetime.now()
        duration = dateutil.relativedelta.relativedelta(toc, tic)
        runtime = '{days:02d}-{hours:02d}:{mins:02d}:{secs:02d}'
        print_str = 'Runtime ' + runtime.format(days=duration.days, 
                                                hours=duration.hours, 
                                                mins=duration.minutes, 
                                                secs=duration.seconds)
        print(print_str)
        logging.info(print_str)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main([None, '-h'])
    else:
        main(sys.argv)
