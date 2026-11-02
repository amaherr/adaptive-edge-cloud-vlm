# Physical network measurement checkpoint v1

The cloud inference server remains on TCP 5050. An independent `iperf3` 3.16 server listens on cloud WSL TCP 5051, and a TCP echo server listens on 5052. After the cloud switched to Ethernet, Windows portproxy forwards these ports at `192.168.8.74` to cloud WSL at `172.23.49.189`; the old Wi-Fi listeners were removed. Windows Public firewall rules restrict all three inbound ports to the edge Windows Wi-Fi address `192.168.8.71`. Local cloud connection tests passed; they are not physical throughput measurements.

Before every physical network block, verify the cloud Ethernet and edge Wi-Fi addresses and the WSL portproxy target. Record independent edge-to-cloud and cloud-to-edge `iperf3` goodput, RTT, configured shaping, timestamps, and Git/model/dataset provenance. Do not infer achieved bandwidth from `tc` settings or application socket timing.

The first unshaped edge-to-cloud check used `iperf3 -c 192.168.8.70 -p 5051 -t 5` from edge WSL on the Wi-Fi path. Current checks use the Ethernet address `192.168.8.74`; the reverse check uses `-R`. The edge interface, shaping tools, and baseline physical path were verified before starting the frozen block grid.

## Initial unshaped observations

The first 5-second edge-to-cloud test reported 89.5 Mbit/s at the sender but 48.8 Mbit/s at the receiver over an 8.65-second receiver interval. A later 10-second test with two warmup seconds reported 31.8 Mbit/s sender, 31.1 Mbit/s receiver, 1,338 sender TCP retransmissions, and zero transfer during the final six measured seconds. Ten ICMP probes reached the cloud without loss, but RTT ranged from 3.68 ms to 2,482 ms (mean 653.9 ms). Two longer tests attempted while the cloud measurement process had disappeared reset before completion; they are invalid measurements.

The reverse 10-second test reported 49.3 Mbit/s sender, 50.2 Mbit/s receiver, zero sender retransmissions, and three consecutive zero-transfer seconds. The cloud Wi-Fi adapter is a TP-Link 802.11n device on channel 1 with a 150 Mbit/s reported link rate and 100% signal. Cloud-to-edge ICMP received no replies, which may be an edge firewall setting and is not treated as evidence of data-path packet loss.

These observations show that the current unshaped path is not stable enough to treat a single measured goodput or RTT as a baseline. No shaped network block has been accepted, and P5 remains open.

Further isolation found that native Windows ping from the cloud PC to its router `192.168.8.1` lost 6 of 20 packets (30%) and had responses as high as 415 ms. Cloud WSL ping to that router had a 3,554 ms maximum in a separate 20-probe run. Thus the instability is visible before the inference server and WSL portproxy. The cloud uses a TP-Link 802.11n adapter on 2.4 GHz channel 1 at a reported 150 Mbit/s link rate; the edge uses an 802.11ax adapter on 5 GHz channel 36 at 574 Mbit/s with 99% signal. These observations implicate the cloud-side wireless path, though they do not identify a specific radio or router fault. A stable Ethernet or 5 GHz cloud connection is needed before accepting P5 blocks or physical latency profiles. Any IP change requires updating the experiment config, Windows portproxy, and edge-restricted firewall addresses before rerunning connectivity checks.

After the cloud PC connected by Ethernet at `192.168.8.74`, native Windows sent 20 router pings with zero loss and every reply under 1 ms. Cloud WSL sent another 20 with zero loss, 0.714 ms mean RTT and 0.967 ms maximum. These checks show a stable cloud-to-router segment during the short observation.

The first edge-to-cloud-host ping set over Ethernet returned 10/10 replies with 3.034 ms mean and 4.418 ms maximum RTT. An independent reverse `iperf3` run sent from cloud to edge at 440 Mbit/s with zero sender retransmissions; the edge received 449 Mbit/s over its 10 measured seconds. Per-second reverse rates stayed between 424 and 462 Mbit/s. A forward run sent from edge to cloud at 309 Mbit/s with one sender retransmission; the cloud received 302 Mbit/s. Per-second forward rates stayed between 233 and 397 Mbit/s without zero-transfer seconds. These are unshaped baseline observations; repeatability and shaped blocks remain to be validated.

A second unshaped pair received 333 Mbit/s edge-to-cloud and 450 Mbit/s cloud-to-edge, with zero sender retransmissions in either direction. The two forward receiver rates differ by about 10%; reverse differs by less than 1%. Both runs completed without zero-transfer intervals. A machine-readable summary of the user-supplied terminal outputs is archived at `artifacts/network/physical-v1/unshaped_ethernet_v1.json` (ignored by Git). It records the time of transcription because exact test start times were not captured. This is a usable unshaped Ethernet baseline for designing low-bandwidth blocks, but each shaped block still requires independent achieved-rate and RTT verification.

## N1 diagnostic block

Both WSL hosts applied `tc qdisc replace dev eth0 root netem delay 10ms rate 2mbit`; `tc qdisc show dev eth0` confirmed the active settings. Edge-to-cloud receiver goodput was 1.91 Mbit/s and cloud-to-edge receiver goodput was 1.97 Mbit/s, with zero sender retransmissions. Edge ping to cloud Windows averaged 12.579 ms, which includes edge egress delay but ends before cloud WSL egress. A separate one-byte TCP echo probe through WSL on both hosts measured 30 post-warmup samples: 23.333 ms median, 24.280 ms p95, and 29.386 ms maximum. The unshaped edge-to-cloud-host ping mean was 3.034 ms, so the echo result is consistent with roughly 20 ms added round-trip delay. `artifacts/network/physical-v1/N1.json` and `N1_tcp_rtt.json` archive the block summary and raw RTT samples; the exact throughput test start times were not captured.

The independent echo probe is `scripts/tcp_rtt_probe.py`. It now has a dedicated cloud Ethernet listener on TCP 5052, forwarded to WSL and restricted by Windows Public firewall to the edge IP. `iperf3` remains on TCP 5051. This permits concurrent rate and RTT validation for later blocks. N1 is a validated diagnostic block; repeat blocks and the full declared network grid are still pending.

A second N1 echo set on dedicated port 5052 returned 23.386 ms median and 23.896 ms p95 over 10 post-warmup samples, closely matching the first set on port 5051. Both qdiscs were removed afterward and verified to return to `mq`.

## N2 diagnostic block

Both WSL egress queues were set to `rate 5mbit delay 10ms`. Independent `iperf3` receiver goodput was 4.76 Mbit/s edge-to-cloud and 4.98 Mbit/s cloud-to-edge, with zero sender retransmissions. The dedicated echo probe measured 23.735 ms median and 24.835 ms p95 over 10 post-warmup samples. The machine-readable summary is `artifacts/network/physical-v1/N2.json`. N2 meets its diagnostic bandwidth and RTT targets; the edge and cloud queues then advanced to N3.

## N3 diagnostic block

At `rate 10mbit delay 10ms` on both WSL egress queues, receiver goodput was 9.41 Mbit/s edge-to-cloud and 9.83 Mbit/s cloud-to-edge, with zero sender retransmissions. The echo probe measured 23.338 ms median and 23.685 ms p95 over 10 post-warmup samples. The summary is `artifacts/network/physical-v1/N3.json`. N3 meets its diagnostic bandwidth and RTT targets; the queues then advanced to N4.

## N4 diagnostic block

At `rate 25mbit delay 10ms` on both WSL egress queues, receiver goodput was 23.6 Mbit/s edge-to-cloud and 24.4 Mbit/s cloud-to-edge, with zero sender retransmissions. The echo probe measured 22.438 ms median and 22.553 ms p95 over 10 post-warmup samples. The summary is `artifacts/network/physical-v1/N4.json`. N4 meets its diagnostic bandwidth and RTT targets; the queues then advanced to N5.

## N5 diagnostic block

At `rate 50mbit delay 10ms` on both WSL egress queues, receiver goodput was 47.1 Mbit/s edge-to-cloud and 49.2 Mbit/s cloud-to-edge, with zero sender retransmissions. The echo probe measured 22.644 ms median and 22.906 ms p95 over 10 post-warmup samples. The summary is `artifacts/network/physical-v1/N5.json`. N5 meets its diagnostic bandwidth and RTT targets; the queues then advanced to N6.

## N6 diagnostic block

At `rate 100mbit delay 10ms` on both WSL egress queues, receiver goodput was 94.4 Mbit/s edge-to-cloud and 99.1 Mbit/s cloud-to-edge, with zero sender retransmissions. The echo probe measured 22.608 ms median and 23.942 ms p95 over 10 post-warmup samples; one sample reached 29.621 ms. The summary is `artifacts/network/physical-v1/N6.json`. N6 meets its diagnostic bandwidth and RTT targets; the queues then advanced to N7.
