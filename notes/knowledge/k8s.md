## setup
sudo snap install microk8s --channel=19 --classic

sudo apt install -y docker.io
sudo systemctl enable docker.service --now
sudo vi /etc/fstab

sudo swapoff -a

sudo apt install -y apt-transport-https curl

sudo curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add
sudo apt-add-repository "deb http://mirrors.ustc.edu.cn/kubernetes/apt kubernetes-xenial main"

```
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
EOF
```
sudo sysctl --system

sudo apt update
sudo apt install -y kubelet kubeadm kubectl
apt-mark hold kubelet kubeadm kubectl

master: cpu > 2, [mem > 2G]

sudo kubeadm init --image-repository registry.aliyuncs.com/google_containers --pod-network-cidr=10.244.0.0/16

#kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
#kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/k8s-manifests/kube-flannel-rbac.yml

kubectl --kubeconfig .kube/config apply -f kube-flannel.yml
kubectl --kubeconfig .kube/config apply -f kube-flannel-rbac.yml

mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

You should now deploy a pod network to the cluster.
Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
https://kubernetes.io/docs/concepts/cluster-administration/addons/

Then you can join any number of worker nodes by running the following on each as root:

kubeadm join 192.168.0.19:6443 --token 7bqgyt.dm6oege1vvu7ri26 --discovery-token-ca-cert-hash sha256:4e7e3ef3318600a7456e2b446a938b0d45e52297ac79c56f90757653e969e49a

`source <(kubectl completion bash)`

## commonds

```
kubectl create deployment nginx-app --image=nginx
kubectl get deployments
kubectl describe deployment nginx-app
kubectl scale --replicas=2 deployment nginx-app
kubectl create service nodeport nginx --tcp=8001:80
kubectl get svc
kubectl delete deployment nginx-app
```

kubectl run http-web --image=httpd --port=80
kubectl expose pod http-web --name=http-service --port=80 --type=NodePort

kubectl create deployment hostnames --image=in28min/helloworld-rest-api:0.0.1.RELEASE
kubectl expose deployment hostnames --type=LoadBalancer --port=8080
kubectl scale  deployment hostnames --replicas=3

kubectl delete pod hostnames --wait=false
kubectl delete pod hostnames --grace-period=0 --force
kubectl run curl --image=radial/busyboxplus:curl -i --tty
#resume using 'kubectl attach curl -c curl -i -t' command when the pod is running
# nslookup hostnames
# curl hostnames:9002/hello

kubectl autoscale deployment hostnames --max=5 --cpu-percent=70
kubectl set image deployment hostnames *=192.168.0.13:5000/skelix/hostname-app:2
kubectl rollout status deployment hostnames
kubectl roolout history deployment hostnames
kubectl roolout history deployment hostnames --revision=2 # revision details
kubectl rollout undo deployment hostnames-deployment
kubectl rollout undo deployment hostnames-deployment --to-revision=3

kubectl get events --sort-by=.metadata.creationTimestamp

# more a node to maintance
kubectl drain <node>
# move it back
kubectl uncordon <node>

kubectl attach <pod>
kubectl exec -it <pod> -- bash
kubectl label pods <pod> healthy=false

kubectl label node node3 storageType=ssd

add `nodeSelector: {storageType: ssd}` to deployment.yaml
