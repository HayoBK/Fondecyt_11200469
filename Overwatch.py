import sys
import Trigger
import psychopy
from psychopy import prefs
prefs.hardware['audioLib'] = 'ptb'
from psychopy import visual, event
from psychopy.iohub import launchHubServer
import psychopy.sound as sound
from psychtoolbox import audio
audioD = sound.getDevices()
allDev = audio.get_devices
import sounddevice
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
from psychopy.hardware import joystick
import XInput
import triad_openvr as vr
from datetime import datetime
# 100 --> Stop trial
# 10 + N --> Trial N comenzó.
#, Izq,Der,For,Back, Still --> 4,6,8,2,5 (como en NumPad) 

def RelojitoSTR():
    ct = datetime.now()
    ts = str(ct.timestamp())
    ct = str(ct)
    lslcl = str(local_clock())
    return ct,ts,lslcl


PUERTO = True
XBOX = True
VR_ACTIVE = True

fOFF = 'f1'
fPAUSE = 'f12'
fGO_ON = 'f5'
f_FULL_STOP = 'f4'
f_FORCE_START = 'f7'

P_LEFT = 4
P_RIGHT = 6
P_FORWARD = 8
P_BACK = 2
P_STILL = 5
#P_TRial es igual a 100 + Numero de Trial
P_FULLSTOP = 202
P_POSSIBLE_STOP = 201
P_FALSE_STOP = 203
P_GO_ON = 200
P_FORCE_START = 205

LSL_Stream = StreamInfo('Overwatch-Markers','Markers', 5 , 0 ,'string','overwatch-Titan')
# Canales = 5: [ Marcadores Primarios, Marcadores Secundarios, Fecha y Hora, Timestamp Del Computador, Timestamp LSL]
LSL_Stream2 = StreamInfo('Overwatch-VR','Datos VR', 6 , 0, 'float32','overwatch-Titan2')
# Canales = 7 : vx,vy,vz,vroll,vjaw,vpitch
LSL_Stream3 = StreamInfo('Overwatch-Joy','Datos Joystick', 2 , 0, 'float32','overwatch-Titan2')

outlet = StreamOutlet(LSL_Stream)
outlet2 = StreamOutlet(LSL_Stream2)
outlet3 = StreamOutlet(LSL_Stream3)

def Empujar (Signal,Secondary):
    global outlet
    a,b,c = RelojitoSTR()
    outlet.push_sample([Signal,Secondary, a,b,c])

def PunchVR (vx,vy,vz,vroll,vjaw,vpitch):
    global outlet2
    outlet2.push_sample([vx,vy,vz,vroll,vjaw,vpitch])

if VR_ACTIVE:
    v= vr.triad_openvr()
    print(v.print_discovered_objects())

if PUERTO:
    trigger = Trigger.Trigger('COM8')
    EN = 255
    trigger.enable(EN)
    trigger.set(0)

#ACTIVAR REGISTRO DE SONIDO------------------------
Parlante = audio.get_devices(device_index=4.0)
rx_buffer = np.ones((10 ** 6, 2), dtype=np.float32)
global SoundData,dB,MaxSound
SoundData = '0'
dB=0
MaxSound = 0
Umbral_S = 55 #Limite de umbral de sonido!

def ReportSound(outdata, frames: int, time, status):
    global SoundData,dB,MaxSound
    volume_norm = np.linalg.norm(outdata) *10
    if status:
        print(status)

    SoundData= str((int(volume_norm)))
    dB = (int(volume_norm))
    if (int(volume_norm)) > MaxSound:
        MaxSound = int(volume_norm)

MyStream = sounddevice.InputStream(callback=ReportSound)

MyStream.start()

H_Trial = 0
H_Status = 0

io = launchHubServer()
keyboard = io.devices.keyboard

Continue_Loop = True
if XBOX:
    joystick.backend = 'pyglet'
win = visual.Window( size=[600,400],pos=[0,0], fullscr=False, screen = 0,
    winType='pyglet', allowGUI=True, allowStencil=False,
    monitor='testMonitor', color=[1,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
barra_Limite = visual.TextStim(win=win, name='text',
    text='-------',
    font='Open Sans',
    pos=(0.45, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='blue', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
barra_Limite.autoDraw = True
barra_Movil = visual.TextStim(win=win, name='text',
    text='-------',
    font='Open Sans',
    pos=(0.45, -0.25), height=0.05, wrapWidth=None, ori=0.0, 
    color='yellow', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
barra_Movil.autoDraw = True
text_VR = visual.TextStim(win=win, name='text',
    text='VR',
    font='Open Sans',
    pos=(0, -0.32), height=0.05, wrapWidth=None, ori=0.0, 
    color='yellow', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
text_VR.autoDraw = True
text_Trial = visual.TextStim(win=win, name='text',
    text='TRIAL = ',
    font='Open Sans',
    pos=(-0.25, 0.35), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
text_Trial.autoDraw = True
text_Trial = visual.TextStim(win=win, name='text',
    text='Con +/- puede ajustar # del Trial',
    font='Open Sans',
    pos=(0.25, 0.4), height=0.03, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
text_Trial.autoDraw = True
text_Trial_Count = visual.TextStim(win=win, name='text',
    text=str(H_Trial),
    font='Open Sans',
    pos=(0, 0.35), height=0.14, wrapWidth=None, ori=0.0, 
    color='yellow', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
text_Trial_Count.autoDraw = True
text_Joy = visual.TextStim(win=win, name='text',
    text='Joystick',
    font='Open Sans',
    pos=(0, 0.1), height=0.04, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
text_Joy.autoDraw = True

text_Trigger_Text = visual.TextStim(win=win, name='text',
    text='Estado de Overwatch-->',
    font='Open Sans',
    pos=(-0.25, 0.20), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
text_Trigger_Text.autoDraw = True

text_Recording = visual.TextStim(win=win, name='text',
    text='Apretar F12 para bloquear y desbloquear! (negro/activo)',
    font='Open Sans',
    pos=(0, 0.45), height=0.02, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
text_Recording.autoDraw = True

text_Trigger_Status = visual.TextStim(win=win, name='text',
    text=str(H_Status),
    font='Open Sans',
    pos=(0.2, 0.20), height=0.05, wrapWidth=None, ori=0.0, 
    color='yellow', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
text_Trigger_Status.autoDraw = True

text = visual.TextStim(win=win, name='text',
    text='Hola Hola',
    font='Open Sans',
    pos=(0, -0.4), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
text.text = 'Apreta - F1 -  para apagar'
text.autoDraw = True

triggers =0
text_triggers = visual.TextStim(win=win, name='text',
    text=('Numero de señales enviadas = '+ str(triggers)),
    font='Open Sans',
    pos=(0, -0.35), height=0.03, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
text_triggers.autoDraw = False

Kdata = visual.TextStim(win=win, name='text',
    text='Key',
    font='Open Sans',
    pos=(0, 0.0), height=0.03, wrapWidth=None, ori=0.0, 
    color='blue', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
Kdata.autoDraw = True

Sdata = visual.TextStim(win=win, name='text',
    text='Output Sound '+ SoundData,
    font='Open Sans',
    pos=(0, -0.25), height=0.04, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
Sdata.autoDraw = True

MSdata = visual.TextStim(win=win, name='text',
    text='Max Output Sound '+ str(MaxSound),
    font='Open Sans',
    pos=(0, -0.15), height=0.04, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
MSdata.autoDraw = True

Flash = False
Fdata = visual.TextStim(win=win, name='text',
    text='O',
    font='Open Sans',
    pos=(0.25, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='yellow', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
Fdata.autoDraw = False
jy=0
jx=0
Move= 'Still'
still= True

if XBOX:
    print('Funca? ',XInput.get_battery_information(0))
    GamePad = XInput.get_state(0)
    print('Funca? ',XInput.get_thumb_values(GamePad))

R_Status= False
win.color = (0,0,0)
text_Trigger_Status.text = "Bloqueado"
jx=0
jy=0
vx = 0
vy =0
vz =0 
vjaw = 0
vpitch = 0
vroll = 0
if VR_ACTIVE:
    oPos = v.devices["hmd_1"].get_pose_euler()

while Continue_Loop:
    win.flip()
    if VR_ACTIVE:
        vPos = v.devices["hmd_1"].get_pose_euler()
        vx= "{:.2f}".format(vPos[0]-oPos[0])
        vy= "{:.2f}".format(vPos[1]-oPos[1])
        vz= "{:.2f}".format(vPos[2]-oPos[2])
        vroll = "{:.2f}".format(vPos[3]-oPos[3])
        vjaw= "{:.2f}".format(vPos[4]-oPos[4])
        vpitch= "{:.2f}".format(vPos[5]-oPos[5])
        text_VR.text = ('X= ',str(vx),' Y= ',str(vy),' Z= ',str(vz))
        text_VR.text = ('Yaw= ',str(vjaw),' Pitch= ',str(vpitch),' Roll= ',str(vroll))
        PunchVR(vPos[0]-oPos[0],vPos[1]-oPos[1],vPos[2]-oPos[2],vPos[3]-oPos[3],vPos[4]-oPos[4],vPos[5]-oPos[5])
    
    if XBOX:
        eventos = XInput.get_events()
        for eve in eventos:
            if eve.type == 6:
                jx=eve.x
                jy=-eve.y
    
    fjx = "{:.2f}".format(jx)
    fjy = "{:.2f}".format(jy)
    outlet3.push_sample([jx,jy])
    if (jx < -0.5) and (R_Status == True):
        Move = 'Left'
        if PUERTO:
            trigger.set(P_LEFT)
        Empujar('NONE',Move)
        #outlet.push_sample([Move])
        still= False
    if (jx > 0.5) and (R_Status == True):
        Move = 'Right'
        if PUERTO:
            trigger.set(P_RIGHT)
        Empujar('NONE',Move)
        #outlet.push_sample([Move])
        still= False
    if (jy < -0.5) and (R_Status == True):
        Move = 'Forward'
        if PUERTO:
            trigger.set(P_FORWARD)
        #outlet.push_sample([Move])
        Empujar('NONE',Move)
        still= False
    if (jy > 0.5) and (R_Status == True):
        Move = 'Back'
        if PUERTO:
            trigger.set(P_BACK)
        #outlet.push_sample([Move])
        Empujar('NONE',Move)
        still= False
    if (abs(jx) < 0.5) and (abs(jy) <0.5) and (still == False) and (R_Status == True):
        Move = 'Still'
        if PUERTO:
            trigger.set(P_STILL)
        #outlet.push_sample([Move])
        Empujar('NONE',Move)
        still = True
   
    t='Joystick --> X= '+ fjx + ' // Y = ' + fjy + ' Move= ' + Move
    text_Joy.text = t
    Flash=False
    if (H_Status == 0) and (R_Status == True):
        text_Trigger_Status.text = "En Reposo"
    if H_Status == 1:
        text_Trigger_Status.text = "Preparados..."
    if H_Status == 2:
        text_Trigger_Status.text = "NAVEGANDO"
    if H_Status == 3:
        text_Trigger_Status.text = "Paramos?"
    for e in keyboard.getPresses():
        a,b,c = RelojitoSTR()
        Kdata.text = e.key + ' - '+ str(c) + ' / ' +str(a)

        if (H_Status == 3) and (R_Status == True) and (e.key==f_FULL_STOP):
            win.color = (1,0,0)
            H_Status = 0
            if PUERTO:
                trigger.set(P_FULLSTOP)
            
            Empujar('Stop confirmado','NONE')
            text_Trigger_Status.text = str(H_Status)            
            text_Trial_Count.text = str(H_Trial)
        if (H_Status == 3) and (R_Status == True) and ((e.key=='return') or (e.key==fGO_ON)):
            if PUERTO:
                trigger.set(P_FALSE_STOP) #Falsa parada
            Empujar('Falso Stop','NONE')
            H_Status = 2
            win.color = (0,0.5,0)
        if (H_Status == 0) and (R_Status == True) and (e.key==fGO_ON):
            Empujar('Partida Fozada','NONE')
            if PUERTO:
                trigger.set(P_FORCE_START) #Partida Forzada
            H_Status = 2
            win.color = (0,0.5,0)
        if (e.key == f_FORCE_START):
            H_Trial +=1 
            H_Status = 2
            R_Status= True
            text_Trigger_Status.text = str(H_Status)            
            text_Trial_Count.text = str(H_Trial)
            if PUERTO:
                trigger.set(100+H_Trial)
            Empujar(str(H_Trial),'NONE')
            win.color = (0,0.5,0)
            if PUERTO:
                trigger.set(P_FORCE_START) #Partida Forzada
            Empujar('Partida Forzada','NONE')
        if (e.key == 'return') and (H_Status < 2) and (R_Status == True): 
            Flash = True
            H_Status +=1
            if H_Status == 1: #Amarillo, preparación
                win.color = ((153/255),(153/255),0)
                H_Trial +=1 
            if H_Status == 2: #Verde, se inició el Trial
                if PUERTO:
                    trigger.set(100+H_Trial)
                Empujar(str(H_Trial),'NONE')
                win.color = (0,0.5,0)
            text_Trigger_Status.text = str(H_Status)
            text_Trial_Count.text = str(H_Trial)
        if (e.key == f_FULL_STOP) and (H_Status > 0) and (R_Status == True):
            H_Status = 0
            if PUERTO:
                trigger.set(P_POSSIBLE_STOP)
            Empujar('Stop','NONE')
            win.color = (1,0,0)
            if PUERTO:
                trigger.set(P_FULLSTOP)
            Empujar('Stop confirmado','NONE')
            text_Trigger_Status.text = str(H_Status)            
            text_Trial_Count.text = str(H_Trial)
        if (e.key == '+'):
            H_Trial +=1
            text_Trial_Count.text = str(H_Trial)
        if (e.key == '-'):
            H_Trial -=1
            text_Trial_Count.text = str(H_Trial)
        if e.key == fOFF:
            Continue_Loop = False
        if e.key == fPAUSE:
            if R_Status == True:
                win.color = (0,0,0)
                text_Trigger_Status.text = "Bloqueado"
                R_Status = False
            elif R_Status == False:
                win.color = (1,0,0)
                text_Trigger_Status.text = "En Reposo"
                H_Status = 0
                R_Status = True
                if VR_ACTIVE:
                    oPos = v.devices["hmd_1"].get_pose_euler()
        if e.key == 'f9':
            if VR_ACTIVE:
                oPos = v.devices["hmd_1"].get_pose_euler()
        if e.key == 'f10':
            Umbral_S -= 1
        if e.key == 'f11':
            Umbral_S += 1
            
    if Flash:
        Fdata.text='X'
        triggers+=1
        text_triggers.text=('Numero de señales enviadas = '+ str(triggers))
    else:
        Fdata.text='O'
    Sdata.text = 'Output Sound '+ SoundData
    MSdata.text = 'Max Output Sound '+ str(round(MaxSound))
    MaxSound-=0.1
    if (H_Status == 2) and (dB > Umbral_S) and (R_Status == True):
        H_Status = 3
        if PUERTO:
            trigger.set(P_POSSIBLE_STOP)
        Empujar('Stop','NONE')
        win.color = ((207/255),(52/255),(118/255))
        text_Trigger_Status.text = str(H_Status)            
        text_Trial_Count.text = str(H_Trial)
    barra_Limite.pos = (0.45, ((Umbral_S/150) - (0.45)))
    barra_Limite.text = '----- ' + str(Umbral_S)
    barra_Movil.pos = (0.45, ((dB/150) - (0.45)))
    barra_Movil.text = '------ ' + SoundData

    io.clearEvents('all')
if PUERTO:
    trigger.disable(EN)

MyStream.stop()