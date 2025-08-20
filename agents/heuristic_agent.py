import numpy as np

ACTION_NAMES = {"SUBNET_SCAN": "subnet_scan", "OS_SCAN": "os_scan", "SERVICE_SCAN": "service_scan", "PROCESS_SCAN": "process_scan",
            "EXPLOIT_SSH": "e_ssh",  "EXPLOIT_FTP": "e_ftp", "EXPLOIT_SAMBA": "e_samba", "EXPLOIT_SMTP": "e_smtp", "EXPLOIT_HTTP": "e_http", 
            "PRIVI_ESCA_TOMCAT": "pe_tomcat", "PRIVI_ESCA_DACLSVC": "pe_daclsvc", "PRIVI_ESCA_CRON": "pe_schtask"}

class Heuristic():

    def __init__(self, observation_space_shape, action_space, action_space_n, subnet_count, host_count, max_machines_in_subnet):
        self.observation_shape = observation_space_shape[0]

        self.max_machines_in_subnet = max_machines_in_subnet
        self.subnet_count = subnet_count
        self.host_vector_count = host_count
        self.host_vec_length = int(self.observation_shape / (self.host_vector_count+1))

        self.action_space = action_space
        self.action_n = action_space_n
        
        self.prev_action = None

        self.used_actions = set()

        self.discovered_count = 0

    def select_action(self, action_name, action_target):
        
        for i in range(0, self.action_space.n):
            action = self.action_space.get_action(i)
            if action.name == action_name and action.target == action_target:
                return i
    

    def get_host_data(self, host_vec):
        host_data = {}
        index = self.subnet_count + 1 + self.max_machines_in_subnet

        host_data["subnet"] = int(np.argmax(host_vec[0:self.subnet_count+1]))
        host_data["host"] = int(np.argmax(host_vec[self.subnet_count+1:index]))

        host_data["compromised"] = bool(host_vec[index])
        host_data["reachable"] = bool(host_vec[index+1])
        host_data["discovered"] = bool(host_vec[index+2])
        host_data["value"] = int(host_vec[index+3])
        host_data["discovery_value"] = int(host_vec[index+4])
        host_data["access"] = int(host_vec[index+5])

        if host_vec[index+6] == 1:
            host_data["os"] = "linux"
        elif host_vec[index+7] == 1:
            host_data["os"] = "windows"
         
        host_data["services"] = []
        if host_vec[index+8] == 1:
            host_data["services"].append("ssh")
        
        if host_vec[index+9] == 1:
            host_data["services"].append("ftp")

        if host_vec[index+10] == 1:
            host_data["services"].append("http")

        host_data["processes"] = []
        if host_vec[index+11] == 1:
            host_data["processes"].append("tomcat")

        if host_vec[index+12] == 1:
            host_data["processes"].append("daclsvc")

        return host_data



    def choose_action(self, state):
        hosts, last_action_data = self.parse_state(state)

        discovered_count = sum(host["discovered"] for host in hosts)

        if self.prev_action is not None:
            if last_action_data == 1 or last_action_data == 2 or self.action_space.get_action(self.prev_action).name == "subnet_scan":
                self.used_actions.add(self.prev_action)
            else:
                if discovered_count > self.discovered_count:
                    self.used_actions.clear()
        
        self.discovered_count = discovered_count
        
        action = self.select_action(ACTION_NAMES["SUBNET_SCAN"], (1,0))
        
        # 1. Проверить PE для всех машин
        for host in sorted(hosts, key=lambda h: -h["value"]):

            if host["access"] == 1 and host["value"] > 0:

                if "tomcat" in host["processes"] and host["os"] == "linux":
                    action = self.select_action(ACTION_NAMES["PRIVI_ESCA_TOMCAT"], (host["subnet"], host["host"]))
                    if action not in self.used_actions:
                        self.prev_action = action
                        return action
                    action = self.select_action(ACTION_NAMES["SUBNET_SCAN"], (1,0))

                if "daclsvc" in host["processes"] and host["os"] == "windows":
                    action = self.select_action(ACTION_NAMES["PRIVI_ESCA_DACLSVC"], (host["subnet"], host["host"]))
                    if action not in self.used_actions:
                        self.prev_action = action
                        return action
                    action = self.select_action(ACTION_NAMES["SUBNET_SCAN"], (1,0))


        # 2. Exploit доступных машин
        for host in sorted(hosts, key=lambda h: -h["value"]):
            if host["reachable"] and host["discovered"] and not host["compromised"]:

                if "ssh" in host["services"] and host["os"] == "linux":
                    action = self.select_action(ACTION_NAMES["EXPLOIT_SSH"], (host["subnet"], host["host"]))
                    if action not in self.used_actions:
                        self.prev_action = action
                        return action
                    action = self.select_action(ACTION_NAMES["SUBNET_SCAN"], (1,0))

                if "http" in host["services"]:
                    action = self.select_action(ACTION_NAMES["EXPLOIT_HTTP"], (host["subnet"], host["host"]))
                    if action not in self.used_actions:
                        self.prev_action = action
                        return action
                    action = self.select_action(ACTION_NAMES["SUBNET_SCAN"], (1,0))

                if "ftp" in host["services"] and host["os"] == "windows":
                    action = self.select_action(ACTION_NAMES["EXPLOIT_FTP"], (host["subnet"], host["host"]))
                    if action not in self.used_actions:
                        self.prev_action = action
                        return action
                    action = self.select_action(ACTION_NAMES["SUBNET_SCAN"], (1,0))
        
        
        # 3. Subnet Scan из скомпрометированных
        compromised_hosts = [h for h in hosts if h["compromised"]]
        for host in compromised_hosts:
            if not all(h["discovered"] for h in hosts):
                action = self.select_action(ACTION_NAMES["SUBNET_SCAN"], (host["subnet"], host["host"]))
                if action not in self.used_actions:
                    self.prev_action = action
                    return action
                else:
                    continue
        
        return self.select_action(ACTION_NAMES["SUBNET_SCAN"], (1,0))

   
    def parse_state(self, state):

        data = []

        for subnet_id in range(self.host_vector_count):
            
            start_idx = subnet_id * self.host_vec_length
            host_vec = state[start_idx : start_idx + self.host_vec_length]

            host_data = self.get_host_data(host_vec)
            data.append(host_data)
        
        temp = self.host_vector_count * self.host_vec_length
        last_action_data_temp = state[temp : temp + self.host_vec_length]
        last_action_data = int(np.argmax(last_action_data_temp[0:4]))

        return data, last_action_data