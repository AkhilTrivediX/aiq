# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared authentication utilities for AI-Q blueprint.

This module provides token retrieval and user info utilities that can be used
by any tool or agent.
"""

from .token_cipher import TokenCipher
from .token_cipher import TokenEncryptionConfigError
from .token_cipher import TokenEncryptionError
from .token_cipher import TokenEncryptionInvalidData
from .token_cipher import token_encryption_configured
from .token_store import SqlTokenStore
from .token_store import StoredToken
from .token_store import TokenStore
from .token_store import get_token_store
from .utils import Principal
from .utils import UserInfo
from .utils import clear_token_fetchers
from .utils import decode_unverified_jwt_payload
from .utils import get_auth_token
from .utils import get_current_principal
from .utils import get_current_user_info
from .utils import get_user_info_from_unverified_token
from .utils import get_verified_current_user
from .utils import register_token_fetcher
from .utils import unregister_token_fetcher

__all__ = [
    "Principal",
    "SqlTokenStore",
    "StoredToken",
    "TokenCipher",
    "TokenEncryptionConfigError",
    "TokenEncryptionError",
    "TokenEncryptionInvalidData",
    "TokenStore",
    "UserInfo",
    "clear_token_fetchers",
    "decode_unverified_jwt_payload",
    "get_auth_token",
    "get_current_principal",
    "get_current_user_info",
    "get_token_store",
    "get_user_info_from_unverified_token",
    "get_verified_current_user",
    "register_token_fetcher",
    "token_encryption_configured",
    "unregister_token_fetcher",
]
