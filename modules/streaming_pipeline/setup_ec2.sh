# remove this package to avoid the following error:
#   pip cannot uninstall : "It is a distutils installed project"
apt autoremove -y python3-pyasn1-modules/focal
apt autoremove -y python3-pexpect/focal
apt autoremove -y python3-entrypoints/focal

# pip install pip --upgrade
# pip install pyopenssl --upgrade