#!/usr/bin/env bash
set -e

echo "
alias ll='ls -alF --color=auto'
alias la='ls -A --color=auto'
alias l='ls -CF --color=auto'
alias claude='claude --dangerously-skip-permissions'
" > ~/.bash_aliases
source ~/.bashrc

echo "[post-create] Checking/installing zsh..."
if ! command -v zsh &> /dev/null; then
    echo "[post-create] Installing zsh..."
    apt-get update -qq
    apt-get install -y -qq zsh
fi

if command -v zsh &> /dev/null; then
    # Define aliases
    aliases="
alias ll='ls -alF --color=auto'
alias la='ls -A --color=auto'
alias l='ls -CF --color=auto'
alias claude='claude --dangerously-skip-permissions'
"
    
    # Check if .zshrc exists and contains our aliases
    if [ -f ~/.zshrc ]; then
        if ! grep -q "alias ll='ls -alF --color=auto'" ~/.zshrc 2>/dev/null; then
            echo "$aliases" >> ~/.zshrc
            echo "[post-create] Added aliases to existing ~/.zshrc"
        else
            echo "[post-create] Aliases already exist in ~/.zshrc, skipping"
        fi
    else
        echo "$aliases" > ~/.zshrc
        echo "[post-create] Created ~/.zshrc with aliases"
    fi
    
    echo "[post-create] zsh is now available at: $(which zsh)"
    
    # Attempt to set zsh as default shell
    ZSH_PATH=$(which zsh)
    if [ -n "$ZSH_PATH" ]; then
        # Check if zsh is in /etc/shells, add if not (requires root)
        if ! grep -q "^$ZSH_PATH$" /etc/shells 2>/dev/null; then
            if [ -w /etc/shells ] 2>/dev/null; then
                echo "$ZSH_PATH" >> /etc/shells
                echo "[post-create] Added zsh to /etc/shells"
            else
                echo "[post-create] Warning: Cannot add zsh to /etc/shells (requires root)"
            fi
        fi
        
        # Try to set default shell using chsh
        if command -v chsh &> /dev/null; then
            echo "$ZSH_PATH" | chsh -s "$ZSH_PATH" 2>/dev/null || {
                echo "[post-create] Could not set default shell with chsh (may require password)"
                # Try with sudo if available
                if command -v sudo &> /dev/null; then
                    echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/chsh" > /tmp/chsh_sudo 2>/dev/null || true
                    sudo chsh -s "$ZSH_PATH" "$USER" 2>/dev/null && \
                        echo "[post-create] Successfully set zsh as default shell using sudo" || \
                        echo "[post-create] Could not set default shell with sudo"
                fi
            }
        fi
        
        # Verify if shell was changed
        CURRENT_SHELL=$(getent passwd "$USER" | cut -d: -f7)
        if [ "$CURRENT_SHELL" = "$ZSH_PATH" ]; then
            echo "[post-create] ✓ zsh is now your default shell!"
        else
            echo "[post-create] Default shell is still: $CURRENT_SHELL"
            echo "[post-create] To manually use zsh, run: $ZSH_PATH"
        fi
    fi
fi

echo "[post-create] Installing poetry via pip"
python -m pip install --upgrade pip
python -m pip install poetry

echo "[post-create] Installing Python dependencies (poetry)"
poetry install --with dev --extras proxy

echo "[post-create] Generating Prisma client"
poetry run prisma generate

echo "[post-create] Installing npm dependencies"
cd ui/litellm-dashboard && npm install --no-audit --no-fund
