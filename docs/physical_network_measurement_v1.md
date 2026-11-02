# Physical network measurement checkpoint v1

The cloud inference server remains on TCP 5050. An independent `iperf3` 3.16 server listens on cloud WSL TCP 5051. After the cloud switched to Ethernet, Windows portproxy forwards `192.168.8.74:5050` and `192.168.8.74:5051` to the WSL server at `172.23.49.189`; the old Wi-Fi listeners were removed. Both Windows Public firewall rules restrict inbound traffic to the edge Windows Wi-Fi address `192.168.8.71`. Local cloud connection tests passed; they are not physical throughput measurements.

Before every physical network block, verify the cloud and edge Wi-Fi addresses and the WSL portproxy target. Record independent edge-to-cloud and cloud-to-edge `iperf3` goodput, RTT, configured shaping, timestamps, and Git/model/dataset provenance. Do not infer achieved bandwidth from `tc` settings or application socket timing. No traffic shaping has been applied yet.

The first unshaped edge-to-cloud check used `iperf3 -c 192.168.8.70 -p 5051 -t 5` from edge WSL on the Wi-Fi path. New checks must use the Ethernet address `192.168.8.74`; the reverse check uses `-R`. Use the frozen block grid in the physical validation roadmap only after the edge interface, available shaping tools, and baseline physical path have been verified.

## Initial unshaped observations

The first 5-second edge-to-cloud test reported 89.5 Mbit/s at the sender but 48.8 Mbit/s at the receiver over an 8.65-second receiver interval. A later 10-second test with two warmup seconds reported 31.8 Mbit/s sender, 31.1 Mbit/s receiver, 1,338 sender TCP retransmissions, and zero transfer during the final six measured seconds. Ten ICMP probes reached the cloud without loss, but RTT ranged from 3.68 ms to 2,482 ms (mean 653.9 ms). Two longer tests attempted while the cloud measurement process had disappeared reset before completion; they are invalid measurements.

The reverse 10-second test reported 49.3 Mbit/s sender, 50.2 Mbit/s receiver, zero sender retransmissions, and three consecutive zero-transfer seconds. The cloud Wi-Fi adapter is a TP-Link 802.11n device on channel 1 with a 150 Mbit/s reported link rate and 100% signal. Cloud-to-edge ICMP received no replies, which may be an edge firewall setting and is not treated as evidence of data-path packet loss.

These observations show that the current unshaped path is not stable enough to treat a single measured goodput or RTT as a baseline. No shaped network block has been accepted, and P5 remains open.

Further isolation found that native Windows ping from the cloud PC to its router `192.168.8.1` lost 6 of 20 packets (30%) and had responses as high as 415 ms. Cloud WSL ping to that router had a 3,554 ms maximum in a separate 20-probe run. Thus the instability is visible before the inference server and WSL portproxy. The cloud uses a TP-Link 802.11n adapter on 2.4 GHz channel 1 at a reported 150 Mbit/s link rate; the edge uses an 802.11ax adapter on 5 GHz channel 36 at 574 Mbit/s with 99% signal. These observations implicate the cloud-side wireless path, though they do not identify a specific radio or router fault. A stable Ethernet or 5 GHz cloud connection is needed before accepting P5 blocks or physical latency profiles. Any IP change requires updating the experiment config, Windows portproxy, and edge-restricted firewall addresses before rerunning connectivity checks.

After the cloud PC connected by Ethernet at `192.168.8.74`, native Windows sent 20 router pings with zero loss and every reply under 1 ms. Cloud WSL sent another 20 with zero loss, 0.714 ms mean RTT and 0.967 ms maximum. These checks show a stable cloud-to-router segment during the short observation. Edge-to-cloud throughput and RTT over the new Ethernet destination remain to be measured before closing P5.
