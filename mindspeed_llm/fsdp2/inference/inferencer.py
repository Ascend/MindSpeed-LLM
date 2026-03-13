import torch.distributed as dist

from mindspeed_llm.fsdp2.utils.logging import get_logger
from .chat_model import ChatModel


logger = get_logger(__name__)


class Inferencer:
    """
    Executes inference tasks, backed by an extensible ChatModel engine.
    """
    def __init__(self, model, tokenizer, args):
        self.chat_model = ChatModel(model, tokenizer, args)
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def run_interactive_chat(self):
        """
        Starts an interactive chat loop (CLI Demo) with streaming support.
        """
        logger.info_rank0(">>> Entering Interactive Chat Mode. Type 'exit' to quit.")
        
        history = []
        
        # --- Loop Invariants ---
        # Extracted outside the while loop to avoid repeated memory allocation
        EXIT_COMMANDS = ("exit", "quit")
        ROLE_USER, ROLE_ASSISTANT = ("user", "assistant")
        ASSISTANT_PREFIX = "Assistant: "
        VISUAL_SEPARATOR = "\n" + "-" * 40

        while True:
            # 1. Sync input across all FSDP ranks to prevent deadlocks
            user_input = self._get_sync_input()

            if user_input.lower() in EXIT_COMMANDS:
                break
            
            if not user_input:
                continue

            history.append({"role": ROLE_USER, "content": user_input})
            
            if self.rank == 0:
                print(ASSISTANT_PREFIX, end="", flush=True)

            response = ""
            
            # 2. Use the streaming API for the typewriter effect
            for new_text in self.chat_model.stream_chat(history):
                if self.rank == 0:
                    print(new_text, end="", flush=True)
                response += new_text

            # Print a visual separator after the response finishes
            logger.info_plain_rank0(VISUAL_SEPARATOR)
            
            history.append({"role": ROLE_ASSISTANT, "content": response})

    def _get_sync_input(self):
        """
        In a multi-card environment, only Rank 0 can receive keyboard input.
        The input must be broadcast to other Ranks to ensure all Ranks receive 
        the exact same Prompt for synchronous collective communication.
        """
        if self.rank == 0:
            try:
                user_input = input("\nUser: ").strip()
            except EOFError:
                user_input = "exit"
        else:
            user_input = None

        if dist.is_initialized():
            objects = [user_input]
            dist.broadcast_object_list(objects, src=0)
            user_input = objects[0]
        
        return user_input