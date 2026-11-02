# Physical network measurement checkpoint v1

The cloud inference server remains on TCP 5050. An independent `iperf3` 3.16 server listens on cloud WSL TCP 5051. Windows portproxy forwards `192.168.8.70:5051` to `172.23.49.189:5051`. The Windows Public firewall rule `Adaptive VLM Iperf3 5051` allows inbound TCP 5051 only from the edge Windows Wi-Fi address `192.168.8.71`. The local cloud connection test passed; it is not a physical throughput measurement.

Before every physical network block, verify the cloud and edge Wi-Fi addresses and the WSL portproxy target. Record independent edge-to-cloud and cloud-to-edge `iperf3` goodput, RTT, configured shaping, timestamps, and Git/model/dataset provenance. Do not infer achieved bandwidth from `tc` settings or application socket timing. No traffic shaping has been applied yet.

The first unshaped edge-to-cloud check is `iperf3 -c 192.168.8.70 -p 5051 -t 5` from edge WSL. The reverse check uses `-R`. Use the frozen block grid in the physical validation roadmap only after the edge interface, available shaping tools, and baseline physical path have been verified.
