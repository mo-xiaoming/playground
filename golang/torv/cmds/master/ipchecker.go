package main

import (
	"github.com/mo-xiaoming/torv/internal/geo"
	"log"
	"sync"
	"time"
)

type clientIPAliveness struct {
	ip string
	lastPing time.Time
	g geo.GeoLocation
}

type clientIPPool struct {
	pool map[string]clientIPAliveness
}

func (ci *clientIPPool) get(ip string) (geo.GeoLocation, error) {
	log.Printf("trying to get geo for %s", ip)
	var g geo.GeoLocation
	var ok bool
	var err error
	if _, ok = ci.pool[ip]; !ok {
		if g, err = geo.IP2GeoLocation(ip); err != nil {
			return geo.GeoLocation{}, err
		}
		log.Printf("got geo %v for ip %s", g, ip)
		ci.pool[ip] = clientIPAliveness{ip: ip, lastPing: time.Now(), g: g}
	}
	return ci.pool[ip].g, nil
}

type aliveness struct {
	lastPing time.Time
	geo geo.GeoLocation
}

type ipPool struct {
	ips       map[string]aliveness
	newIPChan chan string
	ipRemoved chan string
	done      chan struct{}
	m         sync.Mutex
}

func (ipp *ipPool) incomingIP(ip string) {
	ipp.newIPChan <- ip
}

func (ipp *ipPool) removeIP(ip string) {
	ipp.ipRemoved <- ip
}

func (ipp *ipPool) start() {
	const expireDuration = 10 * time.Second
	staleIPCleaner := time.NewTicker(expireDuration)
	defer staleIPCleaner.Stop()

	log.Print("ipchecker started")
	for {
		select {
		case ip := <-ipp.newIPChan:
			ipp.m.Lock()
			if aln, ok := ipp.ips[ip]; !ok {
				if g, err := geo.IP2GeoLocation(ip); err != nil {
					log.Printf("cannot fetch geo location for %s, wait for next ping", ip)
				} else {
					log.Printf("geo(%v) for ip %s", g, ip)
					ipp.ips[ip] = aliveness{lastPing: time.Now(), geo: g}
				}
			} else {
				aln.lastPing = time.Now()
				ipp.ips[ip] = aln
			}
			ipp.m.Unlock()
			log.Printf("%s updated lastPing", ip)
		case ip := <-ipp.ipRemoved:
			ipp.m.Lock()
			if _, ok := ipp.ips[ip]; ok {
				delete(ipp.ips, ip)
				log.Printf("%s offline", ip)
			}
			log.Printf("%d nodes left", len(ipp.ips))
			ipp.m.Unlock()
		case <-staleIPCleaner.C:
			ipp.m.Lock()
			for ip, aln := range ipp.ips {
				if aln.lastPing.Add(expireDuration).Before(time.Now()) {
					log.Printf("%s expired, last seen at %v, now is %v", ip, aln.lastPing, time.Now())
					delete(ipp.ips, ip)
				}
			}
			ipp.m.Unlock()
		case <-ipp.done:
			log.Print("ipchecker exiting")
			break
		}
	}
}

func (ipp *ipPool) closedIP(g geo.GeoLocation) string {
	ipp.m.Lock()
	defer ipp.m.Unlock()

	var min = 40000.0
	var closetIP = ""
	for ip, v := range ipp.ips {
		var newMin = geo.Distance(v.geo, g)
		log.Printf("%v <-> %v is %f", v.geo, g, newMin)
		if min > newMin {
			closetIP = ip
			min = newMin
		}
	}
	return closetIP
}

func (ipp *ipPool) stop() {
	ipp.done <- struct{}{}
}
