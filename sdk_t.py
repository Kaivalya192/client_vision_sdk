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
        self.results = {}            # Stores the latest result per module.
        self.running_modules = {}    # Keeps track of which modules are running.
        self.response_queue = asyncio.Queue()  # Queue for responses from server.

    async def connect(self):
        """Establish a connection to the server (if not already connected)."""
        if self.websocket is None:
            self.websocket = await websockets.connect(self.server_uri)
            # Start listening for incoming messages in a background task.
            asyncio.create_task(self.listen_for_results())

    async def listen_for_results(self):
        """Listen for and process incoming messages from the server."""
        while True:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                # If an image is returned, decode the base64 image into an OpenCV image.
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

                # Optionally, you can have the server add a "module" field to each message.
                module_name = data.get("module", "unknown")
                self.results[module_name] = data
                # Place the received message into the response queue.
                await self.response_queue.put(data)
            except websockets.ConnectionClosed:
                print("Connection lost. Reconnecting...")
                self.websocket = None
                await self.connect()
                break

    async def send_command(self, command, prompt=None, extra_params=None):
        """
        Build a request, send it over the WebSocket,
        and wait for the server response.
        """
        req = {"command": command}
        if prompt is not None:
            req["prompt"] = prompt
        if extra_params:
            req.update(extra_params)

        await self.connect()
        await self.websocket.send(json.dumps(req))
        # Wait for the response to this command.
        data = await self.response_queue.get()
        return data

    async def load_module(self, module_name):
        """
        Loads the module on the server (e.g., sending "obj_detection_load").
        This will initialize the module and load its weights.
        """
        command = f"{module_name}_load"
        response = await self.send_command(command)
        return response

    async def run_module(self, module_name, prompt=None):
        """
        Runs the loaded module on the server (e.g., sending "obj_detection_run").
        For one-shot inference (like Grounding-DINO, OwlViT, LLava) the result is
        returned immediately; for continuous tasks (like YOLO, human pose), the server
        will publish results asynchronously.
        """
        command = f"{module_name}_run"
        response = await self.send_command(command, prompt=prompt)
        # Mark this module as running (for continuous tasks).
        self.running_modules[module_name] = True
        return response

    async def stop_module(self, module_name):
        """
        Stops a continuous module on the server (e.g., sending "obj_detection_stop").
        """
        command = f"{module_name}_stop"
        response = await self.send_command(command)
        if module_name in self.running_modules:
            del self.running_modules[module_name]
        return response

    async def get_current_result(self, module_name):
        """Retrieve the latest result received for a given module."""
        return self.results.get(module_name, None)

    async def close(self):
        """Close the websocket connection."""
        if self.websocket:
            await self.websocket.close()


# A synchronous wrapper around the asynchronous client.
class SyncRemoteInferenceClient:
    """
    This wrapper allows users to work with the remote inference client
    in a blocking, synchronous fashion.
    """
    def __init__(self, server_uri="ws://<server-ip>:8765"):
        self._client = RemoteInferenceClient(server_uri)
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def load_module(self, module_name):
        """
        Loads a module (e.g., "obj_detection", "gdino", "hum_pose", etc.)
        on the server.
        """
        return self._loop.run_until_complete(self._client.load_module(module_name))

    def run_module(self, module_name, prompt=None):
        """
        Runs the previously loaded module on the server.
        For one-shot inferences, you can pass a prompt.
        """
        return self._loop.run_until_complete(self._client.run_module(module_name, prompt))

    def stop_module(self, module_name):
        """Stops a running module on the server."""
        return self._loop.run_until_complete(self._client.stop_module(module_name))

    def get_current_result(self, module_name):
        """
        Returns the latest result (if any) for the given module.
        For continuous modules, this may be updated asynchronously.
        """
        return self._loop.run_until_complete(self._client.get_current_result(module_name))

    def close(self):
        """Close the connection to the server."""
        self._loop.run_until_complete(self._client.close())
