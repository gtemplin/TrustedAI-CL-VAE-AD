#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("rtsp_ip", type=str, help="RTSP Hostname")
    parser.add_argument("--rtsp-port", "-p", type=int, default=554, help="RTSP Port")
    parser.add_argument("--username", "-u", type=str, default=None, help="RTSP access username")
    parser.add_argument("--password", "-s", type=str, default=None, help="RTSP access password")

    args = parser.parse_args()

    rtsp_url = f"{args.rtsp_ip}:{args.rtsp_port}"
    if args.username is not None and args.password is not None:
        rtsp_url = f"{args.username}:{args.password}@{rtsp_url}"
    rtsp_url = f"rtsp://{rtsp_url}"

    cap = cv2.VideoCapture(rtsp_url)

    try:
        while True:
            ret, frame = cap.read()

            frame = cv2.resize(frame, (300, 224))
            cv2.imshow(f"{args.rtsp_ip}:{args.rtsp_port}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()