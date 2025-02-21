import asyncio
import json
import websockets
import cv2
import base64
import numpy as np

class RemoteInferenceClient:
    def __init__(self, server_uri="ws://<server-ip>:8765"):
        self.server_uri = server_uri
        self.websocket = None
        self.results = {}
        self.running_modules = {}
        self.response_queue = asyncio.Queue()  # Queue for responses

    async def connect(self):
        # Only connect if there is no active connection.
        if self.websocket is None:
            self.websocket = await websockets.connect(self.server_uri)
            # Start listening for incoming messages in a background task.
            asyncio.create_task(self.listen_for_results())

    async def listen_for_results(self):
        while True:
            try:
                response = await self.websocket.recv()
                data = json.loads(response)
                # Decode image if present.
                if "image" in data:
                    img_base64 = data["image"]
                    try:
                        img_data = base64.b64decode(img_base64)
                        np_arr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        data["image"] = img
                    except Exception as e:
                        print("Error decoding image:", e)
                        data["image"] = None

                module_name = data.get("module", "unknown")
                self.results[module_name] = data
                await self.response_queue.put(data)
            except websockets.ConnectionClosed:
                print("Connection lost. Reconnecting...")
                # Reset the websocket reference so connect() creates a new connection.
                self.websocket = None
                await self.connect()
                break  # Exit this listener; a new one will be created by connect().

    async def start_module(self, module_name):
        await self.connect()
        if module_name in self.running_modules:
            print(f"{module_name} is already running.")
            return
        
        request = {"command": module_name}
        await self.websocket.send(json.dumps(request))
        self.running_modules[module_name] = True

    async def get_current_result(self, module_name):
        return self.results.get(module_name, None)

    async def get_result(self, module_name, prompt=None):
        await self.connect()
        request = {"command": module_name}
        if prompt:
            request["prompt"] = prompt
        await self.websocket.send(json.dumps(request))
        data = await self.response_queue.get()  # Wait for a response.
        return data

    async def stop_module(self, module_name):
        if module_name not in self.running_modules:
            print(f"{module_name} is not running.")
            return
        request = {"command": "stop_detection"}
        await self.websocket.send(json.dumps(request))
        self.running_modules.pop(module_name, None)

    async def close(self):
        if self.websocket:
            await self.websocket.close()

class SyncRemoteInferenceClient:
    """
    A synchronous wrapper around the asynchronous RemoteInferenceClient.
    This allows users to call methods in a blocking, synchronous manner.
    """
    def __init__(self, server_uri="ws://<server-ip>:8765"):
        self._client = RemoteInferenceClient(server_uri)
        # Retrieve or create an event loop.
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def start_module(self, module_name):
        self._loop.run_until_complete(self._client.start_module(module_name))

    def get_current_result(self, module_name):
        return self._loop.run_until_complete(self._client.get_current_result(module_name))

    def get_result(self, module_name, prompt=None):
        return self._loop.run_until_complete(self._client.get_result(module_name, prompt))

    def stop_module(self, module_name):
        self._loop.run_until_complete(self._client.stop_module(module_name))

    def close(self):
        self._loop.run_until_complete(self._client.close())
