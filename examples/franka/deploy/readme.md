### Env support 

Run the command below:

```bash
uv pip install json_numpy uvicorn fastapi
```


### Structure

Deploy use server-client structure. If you want to deploy with high frequency, make sure network communication is great.


### How to create a server

You should start the server on the this computer. See `server_example.py` as reference. 
The server will recive images and language instruction from client, and it should send the action (or action sequences) to client.


```bash 
CUDA_VISIBLE_DEVICES=5 uv run examples/franka/deploy/server_policy.py policy:checkpoint --policy.config=pi0_franka --policy.dir=checkpoints/pi0_franka/bingwen_thu/29999 --host "0.0.0.0" --port 8000


```


---
