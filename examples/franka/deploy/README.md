### Env support 

Run the command below:

```bash
uv pip install json_numpy uvicorn fastapi draccus
```


### Structure

Deploy use server-client structure. If you want to deploy with high frequency, make sure network communication is great.


### How to create a server

You should start the server on the this computer. See `server_example.py` as reference. 
The server will recive images and language instruction from client, and it should send the action (or action sequences) to client.


```bash 
# pi0-fast
CUDA_VISIBLE_DEVICES=6 uv run examples/franka/deploy/server_policy.py --host "0.0.0.0" --port 9876 --default_prompt "test" policy:checkpoint --policy.config="pi0_fast_franka" --policy.dir="/home/bingwen/Documents/arm_ws/TRUE-Bench/third_party/openpi/checkpoints/pi0_fast_franka/bingwen_pi0_fast_franka/29999"

# pi0
CUDA_VISIBLE_DEVICES=6 uv run examples/franka/deploy/server_policy.py --host "0.0.0.0" --port 9876 --default_prompt "test" policy:checkpoint --policy.config="pi0_franka" --policy.dir="/home/bingwen/Documents/arm_ws/TRUE-Bench/third_party/openpi/checkpoints/pi0_franka/bingwen_pi0_franka/29999"

# 0.0.0.0 receive all ip data, 9876 is a port for connection.
```

You should run `ssh wuqiong3 -L 9876:localhost:9876` to start the terminal, and then keep the terminal open, then you can use the local port(9876) to link the remote server(wuqiong3:9876).

`ssh <remote_host_ssh_config> -L <local_port>:<destination_host_ip>:<destination_port>`
