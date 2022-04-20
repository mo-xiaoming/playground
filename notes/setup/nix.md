## nix
```bash
sh <(curl -L https://nixos.org/nix/install) --daemon

mkdir -p $HOME/.config/nix
echo "experimental-features = nix-command flakes" >> $HOME/.config/nix/nix.conf
```
